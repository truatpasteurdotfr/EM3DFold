from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from em3dfold.scunet.scunet import SCUNet as ReferenceSCUNet
from em3dfold.scunet.scunet_flash_attn import SCUNet as FlashSCUNet


def _parse_config(config_text: str) -> list[int]:
    return [int(x.strip()) for x in config_text.split(",") if x.strip()]


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[dtype_arg]


def _make_model(model_cls, args, device: torch.device, dtype: torch.dtype):
    model = model_cls(
        in_nc=args.in_nc,
        config=args.config,
        dim=args.dim,
        drop_path_rate=args.drop_path_rate,
        input_resolution=args.input_resolution,
        head_dim=args.head_dim,
        window_size=args.window_size,
        n_classes=args.n_classes,
    ).to(device=device)
    if device.type == "cuda":
        model = model.to(dtype=dtype)
    model.eval()
    return model


def _benchmark_model(model, x, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)

    start = time.perf_counter()
    out = None
    for _ in range(iters):
        with torch.inference_mode():
            out = model(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter() - start
    assert out is not None
    return elapsed / iters, out


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SCUNet vs SCUNet flash-attn variant.")
    parser.add_argument("--device", default="7", help="CUDA device id or full torch device string.")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-nc", type=int, default=1)
    parser.add_argument("--n-classes", type=int, default=1)
    parser.add_argument("--input-resolution", type=int, default=48)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--config", type=_parse_config, default="2,2,2,2,2,2,2")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    x = torch.randn(
        args.batch_size,
        args.in_nc,
        args.input_resolution,
        args.input_resolution,
        args.input_resolution,
        device=device,
        dtype=(dtype if device.type == "cuda" else torch.float32),
    )

    ref_model = _make_model(ReferenceSCUNet, args, device, dtype)
    flash_model = _make_model(FlashSCUNet, args, device, dtype)
    flash_model.load_state_dict(ref_model.state_dict(), strict=True)

    ref_time, ref_out = _benchmark_model(ref_model, x, args.warmup, args.iters)
    flash_time, flash_out = _benchmark_model(flash_model, x, args.warmup, args.iters)

    if ref_out.dtype != flash_out.dtype:
        flash_out = flash_out.to(ref_out.dtype)
    max_abs_diff = (ref_out - flash_out).abs().max().item()
    speedup = ref_time / flash_time if flash_time > 0 else float("inf")

    print(f"# device = {device}")
    print(f"# dtype = {dtype}")
    print(f"# batch_size = {args.batch_size}")
    print(f"# input_resolution = {args.input_resolution}")
    print(f"# config = {args.config}")
    print(f"# reference_time_s = {ref_time:.6f}")
    print(f"# flash_time_s = {flash_time:.6f}")
    print(f"# speedup = {speedup:.4f}x")
    print(f"# max_abs_diff = {max_abs_diff:.6e}")


if __name__ == "__main__":
    main()
