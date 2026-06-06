import sys
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
__import__("os").chdir(ROOT_DIR)

import os
import time
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)
from train import TrainDP3Workspace

import tensorrt as trt
trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, namespace="")

from quantize_utils import (
    UNetWrapper,
    TRTInferenceEngine,
    load_trt_engine,
    get_unet_dims,
)


def run_rollout(cfg, policy, mode="fp32", env_runner=None):
    from diffusion_policy_3d.env_runner.base_runner import BaseRunner
    if env_runner is None:
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir="benchmark_rollout_tmp",
        )
    latencies = []
    seed = 100
    if hasattr(env_runner, 'env'):
        env_runner.env.seed(seed)

    policy.cuda().eval()
    original_predict = policy.predict_action

    def timed_predict(obs_dict, **kwargs):
        obs_dict = {k: v.cuda() if torch.is_tensor(v) else v for k, v in obs_dict.items()}
        if mode == "fp16":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                t0 = time.perf_counter()
                result = original_predict(obs_dict, **kwargs)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
        else:
            t0 = time.perf_counter()
            result = original_predict(obs_dict, **kwargs)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
        return result

    policy.predict_action = timed_predict
    runner_log = env_runner.run(policy)
    mem_mb = sum(p.numel() * p.element_size() for p in policy.parameters()) / (1024 ** 2)
    policy.predict_action = original_predict
    lat = np.array(latencies)
    lat_stats = dict(
        mean=float(np.mean(lat)),
        p50=float(np.percentile(lat, 50)),
        p95=float(np.percentile(lat, 95)),
        p99=float(np.percentile(lat, 99)),
        std=float(np.std(lat)),
    )
    return runner_log, lat_stats, mem_mb


def _score(runner_log):
    if runner_log is None:
        return None
    return runner_log.get("test_mean_score",
           runner_log.get("success_rate", None))


def print_result(name, lat_stats, mem_mb, runner_log, baseline_mean=None):
    speedup = (f"  speedup: {baseline_mean / lat_stats['mean']:.2f}x"
               if baseline_mean and lat_stats["mean"] > 0 else "")
    score = _score(runner_log)
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Latency  mean : {lat_stats['mean']:.2f} ms{speedup}")
    print(f"  Latency  std  : {lat_stats['std']:.2f} ms")
    print(f"  Latency  p99  : {lat_stats['p99']:.2f} ms")
    print(f"  GPU Memory    : {mem_mb:.1f} MB")
    if score is not None:
        print(f"  Success rate  : {score:.4f}")
    if runner_log:
        for k, v in runner_log.items():
            if k not in ("test_mean_score", "success_rate") and isinstance(v, float):
                print(f"  {k}: {v:.4f}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    # get checkpoint path from hydra override
    checkpoint = OmegaConf.select(cfg, "checkpoint")
    if checkpoint is None:
        checkpoint = str(pathlib.Path(cfg.hydra.run.dir) / "checkpoints" / "latest.ckpt")
    print(f"[benchmark] Loading checkpoint from {checkpoint}")

    engine_path = str(pathlib.Path(checkpoint).parent / "unet_int8.engine")
    print(f"[benchmark] TRT engine path: {engine_path}")

    results = {}
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir="benchmark_rollout_tmp",
    )

    # ── FP32 ─────────────────────────────────────────────────────────────────
    print("\n[benchmark] === FP32 rollout ===")
    workspace_fp32 = TrainDP3Workspace(cfg, output_dir="benchmark_rollout_tmp")
    workspace_fp32.load_checkpoint(path=checkpoint)
    policy_fp32 = workspace_fp32.ema_model if cfg.training.use_ema else workspace_fp32.model
    policy_fp32.cuda().eval()
    fp32_log, fp32_lat, fp32_mem = run_rollout(cfg, policy_fp32, mode="fp32", env_runner=env_runner)
    print_result("PyTorch FP32 (baseline)", fp32_lat, fp32_mem, fp32_log)
    results["FP32"] = (fp32_lat, fp32_mem, fp32_log)

    # ── FP16 ─────────────────────────────────────────────────────────────────
    print("\n[benchmark] === FP16 rollout ===")
    workspace_fp16 = TrainDP3Workspace(cfg, output_dir="benchmark_rollout_tmp")
    workspace_fp16.load_checkpoint(path=checkpoint)
    policy_fp16 = workspace_fp16.ema_model if cfg.training.use_ema else workspace_fp16.model
    policy_fp16 = policy_fp16.half().cuda().eval()
    policy_fp16 = torch.compile(policy_fp16, mode="reduce-overhead")
    fp16_log, fp16_lat, fp16_mem = run_rollout(cfg, policy_fp16, mode="fp16", env_runner=env_runner)
    print_result("PyTorch FP16", fp16_lat, fp16_mem, fp16_log, baseline_mean=fp32_lat["mean"])
    results["FP16"] = (fp16_lat, fp16_mem, fp16_log)

    # ── INT8 TRT ──────────────────────────────────────────────────────────────
    print(f"\n[benchmark] === INT8 TRT rollout ({engine_path}) ===")
    workspace_int8 = TrainDP3Workspace(cfg, output_dir="benchmark_rollout_tmp")
    workspace_int8.load_checkpoint(path=checkpoint)
    policy_int8 = workspace_int8.ema_model if cfg.training.use_ema else workspace_int8.model
    load_trt_engine(engine_path, policy_int8)
    policy_int8.cuda().eval()
    int8_log, int8_lat, int8_mem = run_rollout(cfg, policy_int8, mode="int8", env_runner=env_runner)
    print_result("TensorRT INT8", int8_lat, int8_mem, int8_log, baseline_mean=fp32_lat["mean"])
    results["INT8 (TRT)"] = (int8_lat, int8_mem, int8_log)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'Mode':<20} {'Mean(ms)':>10} {'p95(ms)':>9} "
          f"{'Mem(MB)':>9} {'Speedup':>9} {'SuccessRate':>12}")
    print(f"  {'-' * 71}")
    baseline_mean = fp32_lat["mean"]
    for name, (lat, mem, log) in results.items():
        speedup = f"{baseline_mean / lat['mean']:.2f}x" if lat["mean"] > 0 else "n/a"
        score = _score(log)
        score_s = f"{score:.4f}" if score is not None else "n/a"
        print(f"  {name:<20} {lat['mean']:>10.2f} {lat['p95']:>9.2f} "
              f"{mem:>9.1f} {speedup:>9} {score_s:>12}")
    print("=" * 78)


if __name__ == "__main__":
    main()
