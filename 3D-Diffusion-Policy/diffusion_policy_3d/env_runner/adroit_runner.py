import collections
import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.env import AdroitEnv
from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class AdroitRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 n_overlap=0,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,  # used by MultiStepWrapper for action space shape
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_overlap = n_overlap  # re-query this many steps early; execute n_action_steps - n_overlap per cycle
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        


        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            n_overlap = self.n_overlap
            n_execute = self.n_action_steps - n_overlap
            use_overlap = n_overlap > 0

            # committed: (n_overlap, Da) numpy array of actions already committed for the next
            # chunk. None until after the first prediction.
            committed = None

            while not done:
                # -- build obs tensors --
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(device=device))
                obs_dict_input = {
                    'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                    'agent_pos':   obs_dict['agent_pos'].unsqueeze(0),
                }

                # Pass the committed actions as leftover so the model inpaints them and
                # predicts everything else fresh: output = [a7, a8, b9, b10, ..., b(T-1)]
                leftover_tensor = None
                if use_overlap and committed is not None:
                    leftover_tensor = torch.from_numpy(committed).to(
                        device=device, dtype=obs_dict['agent_pos'].dtype).unsqueeze(0)

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input, leftover_actions=leftover_tensor)

                # output shape: (T - action_start, Da)
                # = [committed(n_overlap), fresh(rest)]
                output = action_dict['action'].detach().cpu().numpy().squeeze(0)

                # n_fresh: how many fresh actions to execute this cycle.
                # Comes from ChunkSizePredictor when use_dynamic_chunk=True, else fixed n_execute.
                n_fresh = action_dict.get('n_fresh') or n_execute

                if use_overlap and committed is not None:
                    # [committed(n_overlap) | fresh(rest)] — execute committed + n_fresh fresh
                    fresh = output[n_overlap:]
                    n_fresh = min(n_fresh, len(fresh))
                    chunk = np.concatenate([committed, fresh[:n_fresh]], axis=0)
                    # next committed = the n_overlap actions immediately after this chunk
                    committed = fresh[n_fresh:n_fresh + n_overlap]
                elif use_overlap:
                    # First cycle: execute n_fresh; carry next n_overlap as committed
                    n_fresh = min(n_fresh, len(output) - n_overlap)
                    chunk = output[:n_fresh]
                    committed = output[n_fresh:n_fresh + n_overlap]
                else:
                    chunk = output[:n_fresh]

                obs, reward, done, info = env.step(chunk)
                num_goal_achieved += np.sum(info['goal_achieved'])
                done = np.all(done)
                actual_step_count += 1

            all_success_rates.append(info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data
