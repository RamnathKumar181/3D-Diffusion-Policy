import argparse
import os
import numpy as np
import torch
import torch.utils.data
import tensorrt as trt
import dill
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="") 

class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,       # (B, horizon, action_dim)
        timestep: torch.Tensor,     # (B,) float32
        global_cond: torch.Tensor,  # (B, global_cond_dim)
    ) -> torch.Tensor:
        return self.unet(sample=sample, timestep=timestep,
                         local_cond=None, global_cond=global_cond)


class UNetCalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, action_dim, horizon, global_cond_dim, n_samples=200):
        self.action_dim      = action_dim
        self.horizon         = horizon
        self.global_cond_dim = global_cond_dim
        self.n_samples       = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample      = torch.clamp(torch.randn(self.horizon, self.action_dim), -2.0, 2.0)
        timestep    = torch.tensor(float(idx % 100))
        global_cond = torch.randn(self.global_cond_dim) * 0.5
        return sample, timestep, global_cond

class UNetInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, cache_file, action_dim, horizon, global_cond_dim):
        super().__init__()
        self.cache_file      = cache_file
        self.dataloader      = iter(dataloader)
        self.batch_size      = dataloader.batch_size
        self.action_dim      = action_dim
        self.horizon         = horizon
        self.global_cond_dim = global_cond_dim

        # pre-allocate persistent CUDA tensors — TRT reads from their data_ptr()
        self.d_sample      = torch.zeros(
            self.batch_size, horizon, action_dim,
            dtype=torch.float32, device="cuda").contiguous()
        self.d_timestep    = torch.zeros(
            self.batch_size,
            dtype=torch.float32, device="cuda").contiguous()
        self.d_global_cond = torch.zeros(
            self.batch_size, global_cond_dim,
            dtype=torch.float32, device="cuda").contiguous()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            sample, timestep, global_cond = next(self.dataloader)
        except StopIteration:
            return None

        # copy into persistent GPU buffers
        self.d_sample.copy_(sample.float())
        self.d_timestep.copy_(timestep.float().reshape(self.batch_size))
        self.d_global_cond.copy_(global_cond.float())

        # return raw GPU pointers — TRT reads directly from these
        return [
            int(self.d_sample.data_ptr()),
            int(self.d_timestep.data_ptr()),
            int(self.d_global_cond.data_ptr()),
        ]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[calibrator] Reading cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"[calibrator] Writing cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def get_unet_dims(policy):
    return dict(
        action_dim      = policy.action_dim,
        horizon         = policy.horizon,
        global_cond_dim = policy.obs_feature_dim * policy.n_obs_steps
                          if policy.obs_as_global_cond else policy.obs_feature_dim,
    )

def export_onnx(policy, onnx_path, batch_size=1):
    dims = get_unet_dims(policy)
    action_dim      = dims["action_dim"]
    horizon         = dims["horizon"]
    global_cond_dim = dims["global_cond_dim"]

    unet = UNetWrapper(policy.model).cuda().eval()

    dummy_sample      = torch.randn(batch_size, horizon, action_dim).cuda()
    dummy_timestep    = torch.zeros(batch_size).cuda()
    dummy_global_cond = torch.randn(batch_size, global_cond_dim).cuda()

    print(f"[export_onnx] Exporting to {onnx_path}...")
    torch.onnx.export(
        unet,
        (dummy_sample, dummy_timestep, dummy_global_cond),
        onnx_path,
        input_names=["sample", "timestep", "global_cond"],
        output_names=["noise_pred"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[export_onnx] Saved to {onnx_path}")
    return dims


def build_engine(onnx_path, engine_path, dims, calibrator, batch_size=1):
    action_dim      = dims["action_dim"]
    horizon         = dims["horizon"]
    global_cond_dim = dims["global_cond_dim"]

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[build_engine] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  

    # enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # allow fp16 fallback
    config.int8_calibrator = calibrator

    # set input shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("sample",      
        (batch_size, horizon, action_dim),
        (batch_size, horizon, action_dim),
        (batch_size, horizon, action_dim))
    profile.set_shape("timestep",    
        (batch_size,), (batch_size,), (batch_size,))
    profile.set_shape("global_cond", 
        (batch_size, global_cond_dim),
        (batch_size, global_cond_dim),
        (batch_size, global_cond_dim))
    config.add_optimization_profile(profile)

    print("[build_engine] Building INT8 TensorRT engine (may take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build_engine] Saved engine to {engine_path}")


class TRTInferenceEngine:
    """Wraps a TensorRT engine for inference — no pycuda needed."""
    def __init__(self, engine_path):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        print(f"[TRTInferenceEngine] Loaded engine from {engine_path}")

    def infer(self, sample, timestep, global_cond):
        """All inputs: torch cuda tensors. Returns torch cuda tensor."""
        # Pass 1: set all input shapes first
        self.context.set_input_shape("sample", tuple(sample.shape))
        self.context.set_input_shape("timestep", tuple(timestep.shape))
        self.context.set_input_shape("global_cond", tuple(global_cond.shape))
         
        # Pass 2: set all tensor addresses (inputs + output)
        self.context.set_tensor_address("sample", sample.data_ptr())
        self.context.set_tensor_address("timestep", timestep.data_ptr())
        self.context.set_tensor_address("global_cond", global_cond.data_ptr())
    
        # Output shape only valid after all input shapes resolved
        out_shape = self.context.get_tensor_shape("noise_pred")
        output = torch.zeros(*out_shape, dtype=torch.float32, device="cuda").contiguous()
        self.context.set_tensor_address("noise_pred", output.data_ptr())
    
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        
        return output


# ---------------------------------------------------------------------------
# Patch policy.model with TRT engine for eval
# ---------------------------------------------------------------------------
def load_trt_engine(engine_path, policy):
    """
    Replaces policy.model with TRT engine.
    Call after self.load_checkpoint() in train.py eval().

    Usagage        from quantize_dp3_onnx import load_trt_engine
        load_trt_engine("unet_int8.engine", policy)
    """
    trt_engine = TRTInferenceEngine(engine_path)

    class TRTUnetModule(torch.nn.Module):
        def __init__(self, engine):
            super().__init__()
            self._engine = engine

        def forward(self, sample, timestep,
                    local_cond=None, global_cond=None, **kwargs):
              
            sample = sample.float().contiguous()
            timestep = timestep.float().cuda()
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)  # () -> (1,)
            timestep = timestep.contiguous()
            
            global_cond = global_cond.float().contiguous()
            return self._engine.infer(
                sample.float(),
                timestep.float(),
                global_cond.float(),
            )

    policy.model = TRTUnetModule(trt_engine)
    print(f"[load_trt_engine] policy.model replaced with TRT INT8 engine")


def load_policy_from_checkpoint(ckpt_path):
    import hydra
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg     = payload["cfg"]
    model   = hydra.utils.instantiate(cfg.policy)
    keys    = payload["state_dicts"]

    if "model" in keys:
        model.load_state_dict(keys["model"])
    elif "ema_model" in keys:
        model.load_state_dict(keys["ema_model"])
    else:
        raise KeyError(f"No model key. Available: {list(keys.keys())}")

    print(f"[load_policy] Loaded from {ckpt_path}")
    return model

if __name__ == "__main__":
    import sys, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.insert(0, ROOT_DIR)
    __import__("os").chdir(ROOT_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        required=True)
    parser.add_argument("--onnx",              default=None)
    parser.add_argument("--engine",            default=None)
    parser.add_argument("--cache-file",        default=None)
    parser.add_argument("--calibration-steps", type=int, default=200)
    parser.add_argument("--batch-size",        type=int, default=1)
    args = parser.parse_args()
    
    ckpt_dir = str(pathlib.Path(args.checkpoint).parent)
    if args.onnx       is None: 
        args.onnx       = f"{ckpt_dir}/unet.onnx"
    if args.engine     is None: 
        args.engine     = f"{ckpt_dir}/unet_int8.engine"
    if args.cache_file is None: 
        args.cache_file = f"{ckpt_dir}/calibration.cache"

    policy = load_policy_from_checkpoint(args.checkpoint)
    policy.cuda().eval()

    # Step 1: export ONNX
    dims = export_onnx(policy, args.onnx, args.batch_size)

    # Step 2: build calibrator
    calib_dataset = UNetCalibrationDataset(
        action_dim=dims["action_dim"],
        horizon=dims["horizon"],
        global_cond_dim=dims["global_cond_dim"],
        n_samples=args.calibration_steps,
    )
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=1,
    )
    calibrator = UNetInt8Calibrator(
        dataloader=calib_loader,
        cache_file=args.cache_file,
        action_dim=dims["action_dim"],
        horizon=dims["horizon"],
        global_cond_dim=dims["global_cond_dim"],
    )

    # Step 3: build TRT engine
    build_engine(args.onnx, args.engine, dims, calibrator, args.batch_size)

    # Step 4: sanity check
    print("[main] Running sanity check...")
    engine = TRTInferenceEngine(args.engine)
    s  = torch.randn(args.batch_size, dims["horizon"], dims["action_dim"]).cuda()
    t  = torch.zeros(args.batch_size).cuda()
    gc = torch.randn(args.batch_size, dims["global_cond_dim"]).cuda()
    out = engine.infer(s, t, gc)
    print(f"[main] Sanity check OK. Output shape: {out.shape}")
    print("[main] Done! Use load_trt_engine() in eval to deploy.")
