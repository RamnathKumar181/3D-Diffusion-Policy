alg_name=${1}
task_name=${2}
addition_info=${3}
seed=${4}
gpu_id=${5}

config_name=${alg_name}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"
ckpt_path="${run_dir}/checkpoints/latest.ckpt"
engine_path="${run_dir}/checkpoints/unet_int8.engine"

cd 3D-Diffusion-Policy
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUDA_LAUNCH_BLOCKING=0

echo -e "\033[33mCheckpoint dir: ${run_dir}/checkpoints/\033[0m"

# Step 1: Build INT8 TRT engine only if it doesn't exist
if [ ! -f "${engine_path}" ]; then
    echo -e "\033[36m[Step 1] INT8 engine not found, building...\033[0m"
    python quantize_utils.py --checkpoint ${ckpt_path}
    echo -e "\033[32m[Step 1] Engine built successfully.\033[0m"
else
    echo -e "\033[33m[Step 1] Engine already exists at ${engine_path}, skipping build.\033[0m"
fi

# Step 2: Run benchmark with Hydra config
echo -e "\033[36m[Step 2] Running benchmark...\033[0m"
python benchmark_utils.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            checkpoint=${ckpt_path} \
                            task.env_runner.eval_episodes=20 \
                            "${@:6}"
echo -e "\033[32m[Step 2] Benchmark complete.\033[0m"
