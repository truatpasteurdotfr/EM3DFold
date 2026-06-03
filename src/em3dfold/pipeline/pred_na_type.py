from __future__ import annotations

import argparse
import importlib
import os
import queue
import random
import threading
import time
import warnings
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from em3dfold.utils.cryo_utils import chunk_generator, get_batch_from_generator, map_batch_to_map, pad_map, parse_map
from em3dfold.utils.misc_utils import abspath
from em3dfold.utils.torch_utils import get_device_names

warnings.filterwarnings("ignore")


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def normalize_density_map(grid: np.ndarray, percentile: float = 99.999) -> tuple[np.ndarray, float]:
    grid = np.asarray(grid, dtype=np.float32)
    positive_values = grid[grid > 0]
    if positive_values.size > 0:
        maximum = float(np.percentile(positive_values, percentile))
    else:
        maximum = float(np.max(grid))

    if maximum > 0.0:
        normalized = np.clip(grid, 0.0, maximum).astype(np.float32, copy=False) / (maximum + 1e-6)
    else:
        normalized = np.zeros_like(grid, dtype=np.float32)
    return normalized.astype(np.float32, copy=False), maximum


def _gaussian_patch_weight(box_size: int, sigma: float | None = None, dtype=np.float32) -> np.ndarray:
    if sigma is None:
        sigma = float(box_size) / 4.0
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError(f"gaussian sigma must be positive, got {sigma}")

    coords = np.arange(box_size, dtype=np.float32)
    center = (float(box_size) - 1.0) / 2.0
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    dist2 = (zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2
    weight = np.exp(-dist2 / (2.0 * sigma ** 2)).astype(dtype, copy=False)

    weight_min = float(weight.min())
    weight_max = float(weight.max())
    if weight_max > weight_min:
        weight = 1.0 + 2.0 * (weight - weight_min) / (weight_max - weight_min)
    else:
        weight = np.ones_like(weight, dtype=dtype)
    return weight.astype(dtype, copy=False)


def _map_batch_to_map_gaussian(
    pred_map: np.ndarray,
    denominator: np.ndarray,
    positions,
    batch: np.ndarray,
    box_size: int,
    patch_weight: np.ndarray,
):
    volume_shape = np.asarray(pred_map.shape[1:], dtype=np.int64)
    patch_weight = np.asarray(patch_weight, dtype=pred_map.dtype)

    for position, chunk in zip(positions, batch):
        start = np.asarray(position, dtype=np.int64)
        end = start + int(box_size)
        dst_start = np.maximum(start, 0)
        dst_end = np.minimum(end, volume_shape)
        if np.any(dst_start >= dst_end):
            continue

        src_start = dst_start - start
        src_end = src_start + (dst_end - dst_start)

        dst_slices = tuple(slice(int(dst_start[i]), int(dst_end[i])) for i in range(3))
        src_slices = tuple(slice(int(src_start[i]), int(src_end[i])) for i in range(3))
        local_weight = patch_weight[src_slices]

        pred_map[(slice(None),) + dst_slices] += chunk[(slice(None),) + src_slices] * local_weight[None, ...]
        denominator[(slice(None),) + dst_slices] += local_weight[None, ...]
    return pred_map, denominator


class _BatchPrefetcher:
    _END = object()

    def __init__(self, generator, batch_size: int, *, dtype=np.float32, max_prefetch: int = 2):
        self.generator = generator
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self._queue = queue.Queue(maxsize=max_prefetch)
        self._worker = None
        self._stop_event = threading.Event()

    def __enter__(self):
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join()
            self._worker = None

    def _put_until_stopped(self, item):
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self):
        try:
            while not self._stop_event.is_set():
                batch = get_batch_from_generator(self.generator, self.batch_size, dtype=self.dtype)
                positions, _ = batch
                if len(positions) == 0:
                    self._put_until_stopped(self._END)
                    return
                self._put_until_stopped(batch)
        except Exception as exc:
            self._put_until_stopped(exc)

    def get(self):
        item = self._queue.get()
        if item is self._END:
            return [], np.zeros((0,), dtype=self.dtype)
        if isinstance(item, Exception):
            raise item
        return item


def _strip_state_dict_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            output[key[len("model."):]] = value
        elif key.startswith("module."):
            output[key[len("module."):]] = value
        else:
            output[key] = value
    return output


def _find_config_snapshot(ckpt_path: Path) -> Path | None:
    candidates = [
        ckpt_path.parent / "config_snapshot.yaml",
        ckpt_path.parent.parent / "config_snapshot.yaml",
        ckpt_path.parent.parent.parent / "config_snapshot.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_model_kwargs_from_snapshot(snapshot_path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except Exception:
        return None

    data = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "model" not in data:
        return None
    model_cfg = data["model"]
    return {
        "module": str(model_cfg["module"]),
        "class_name": str(model_cfg["class_name"]),
        "in_nc": int(model_cfg.get("in_nc", 1)),
        "config": list(model_cfg.get("config", [2, 2, 2, 2, 2, 2, 2])),
        "dim": int(model_cfg.get("dim", 32)),
        "drop_path_rate": float(model_cfg.get("drop_path_rate", 0.0)),
        "input_resolution": int(model_cfg.get("input_resolution", 48)),
        "head_dim": int(model_cfg.get("head_dim", 16)),
        "window_size": int(model_cfg.get("window_size", 3)),
        "n_classes": int(model_cfg.get("n_classes", 4)),
    }


def _count_sequential_children(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    indices = set()
    token = prefix + "."
    for key in state_dict:
        if key.startswith(token):
            rest = key[len(token):]
            first = rest.split(".", 1)[0]
            if first.isdigit():
                indices.add(int(first))
    return len(indices)


def _infer_model_kwargs_from_state_dict(state_dict: dict[str, torch.Tensor], *, box_size: int) -> dict[str, Any]:
    if "m_head.0.weight" not in state_dict or "m_tail.0.weight" not in state_dict:
        raise ValueError("Unable to infer SCUNet architecture from checkpoint state_dict.")

    config = [
        _count_sequential_children(state_dict, "m_down1") - 1,
        _count_sequential_children(state_dict, "m_down2") - 1,
        _count_sequential_children(state_dict, "m_down3") - 1,
        _count_sequential_children(state_dict, "m_body"),
        _count_sequential_children(state_dict, "m_up3") - 1,
        _count_sequential_children(state_dict, "m_up2") - 1,
        _count_sequential_children(state_dict, "m_up1") - 1,
    ]
    return {
        "module": "em3dfold.scunet.scunet",
        "class_name": "SCUNet",
        "in_nc": int(state_dict["m_head.0.weight"].shape[1]),
        "config": config,
        "dim": int(state_dict["m_head.0.weight"].shape[0]),
        "drop_path_rate": 0.0,
        "input_resolution": int(box_size),
        "head_dim": 16,
        "window_size": 3,
        "n_classes": int(state_dict["m_tail.0.weight"].shape[0]),
    }


def _build_model_from_ckpt(ckpt_path: str, *, box_size: int):
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _strip_state_dict_prefix(state_dict)

    model_kwargs = None
    snapshot = _find_config_snapshot(ckpt_path)
    if snapshot is not None:
        model_kwargs = _load_model_kwargs_from_snapshot(snapshot)
    if model_kwargs is None:
        model_kwargs = _infer_model_kwargs_from_state_dict(state_dict, box_size=box_size)

    model_module = importlib.import_module(str(model_kwargs["module"]))
    model_class = getattr(model_module, str(model_kwargs["class_name"]))
    init_kwargs = {key: value for key, value in model_kwargs.items() if key not in {"module", "class_name"}}
    init_kwargs["input_resolution"] = int(box_size)
    model = model_class(**init_kwargs).eval()
    model.load_state_dict(state_dict, strict=True)
    return model, model_kwargs


def _prepare_model_for_devices(model: torch.nn.Module, devices: list[str]) -> tuple[torch.nn.Module, str, int]:
    if not devices:
        raise ValueError("No device is specified.")
    primary_device = devices[0]
    if len(devices) > 1 and all(str(device).startswith("cuda:") for device in devices):
        device_ids = [int(str(device).split(":", 1)[1]) for device in devices]
        model = model.to(primary_device)
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        return model, primary_device, len(device_ids)
    model = model.to(primary_device)
    return model, primary_device, 1


def run_na_type_inference_on_map(
    ckpt_path: str,
    map_file: str,
    *,
    apix: float,
    box_size: int,
    stride: int,
    batch_size: int,
    devices: list[str],
    gaussian_weight: bool,
    gaussian_sigma: float | None,
    fp16: bool,
    normalize_percentile: float,
):
    print(f"# Load map data from {map_file}")
    em_map, origin, nxyz, voxel_size = parse_map(map_file, ignorestart=False, apix=apix)
    print(f"# Map dimensions = {nxyz}")

    norm_map, density_clip_max = normalize_density_map(em_map, percentile=normalize_percentile)
    padded_map = pad_map(norm_map, box_size, dtype=np.float32, padding=0.0)

    raw_model, model_kwargs = _build_model_from_ckpt(ckpt_path, box_size=box_size)
    n_classes = int(model_kwargs.get("n_classes", 4))
    if n_classes != 4:
        raise ValueError(f"Expected a 4-class NA-type model, got n_classes={n_classes}")

    model, primary_device, world_size = _prepare_model_for_devices(raw_model, devices)
    print(f"# Loaded model: {model_kwargs['module']}.{model_kwargs['class_name']}")
    print(f"# Running on devices {devices}")
    print(f"# Per-device batch size = {batch_size}")
    print(f"# Effective batch size = {batch_size * max(world_size, 1)}")
    print(f"# Density clip max = {density_clip_max:.6f}")

    map_pred = np.zeros((n_classes,) + padded_map.shape, dtype=np.float32)
    denominator = np.zeros((n_classes,) + padded_map.shape, dtype=np.float32)

    patch_weight = None
    if gaussian_weight:
        patch_weight = _gaussian_patch_weight(box_size, sigma=gaussian_sigma, dtype=np.float32)
        print(
            "# Using Gaussian patch fusion "
            f"(sigma={float(gaussian_sigma) if gaussian_sigma is not None else box_size / 4.0:.4f}, "
            f"weight_range={patch_weight.min():.4f}-{patch_weight.max():.4f})"
        )
    else:
        print("# Using uniform patch fusion")

    generator = chunk_generator(padded_map, box_size=box_size, stride=stride, pre_scaled=True)
    ncx, ncy, ncz = [ceil(nxyz[2 - i] / stride) for i in range(3)]
    total_steps = float(ncx * ncy * ncz)
    acc_steps, acc_steps_x, l_bar = 0.0, 0, 0
    start_time = time.time()
    effective_batch_size = batch_size * max(world_size, 1)
    amp_dtype = torch.float16 if fp16 and str(primary_device).startswith("cuda") else None

    with _BatchPrefetcher(generator, effective_batch_size, dtype=np.float32) as prefetcher:
        while True:
            positions, chunks = prefetcher.get()
            if len(positions) == 0:
                break

            acc_steps += len(chunks)
            acc_steps_x = int((acc_steps / total_steps) * 100.0) // 5
            if acc_steps_x > l_bar:
                l_bar = acc_steps_x
                elapsed = time.time() - start_time
                bar = f"|{'#' * (2 * l_bar)}{'-' * ((20 - l_bar) * 2)}| {int(l_bar * 5)}% {elapsed:.4f} seconds elapsed"
                print(f"\r{bar}", flush=True)

            x_batch = torch.from_numpy(chunks).view(-1, 1, box_size, box_size, box_size).to(primary_device)
            with torch.no_grad():
                if amp_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        output = model(x_batch)
                else:
                    output = model(x_batch)
            y_pred = output.detach().cpu().numpy().astype(np.float32, copy=False)

            if patch_weight is None:
                map_pred, denominator = map_batch_to_map(map_pred, denominator, positions, y_pred, box_size)
            else:
                map_pred, denominator = _map_batch_to_map_gaussian(
                    map_pred,
                    denominator,
                    positions,
                    y_pred,
                    box_size,
                    patch_weight,
                )

    map_pred = (map_pred / denominator.clip(min=1.0))[
        :,
        box_size : box_size + nxyz[2],
        box_size : box_size + nxyz[1],
        box_size : box_size + nxyz[0],
    ]

    if acc_steps < total_steps:
        print("\r|########################################| 100%", flush=True)

    return map_pred.astype(np.float32, copy=False), origin, voxel_size, model_kwargs, density_clip_max


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--input", "-i", type=str, required=True, help="Input EM density map file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt", "-k", type=str, required=True, help="Lightning checkpoint (.ckpt) for the 4-class NA-type model")
    parser.add_argument("--device", "-g", type=str, default="0", help="Which device(s) to use, e.g. '0', 'cpu', or '0,1'")
    parser.add_argument("--batchsize", "-b", type=int, default=40, help="Per-device batch size for prediction")
    parser.add_argument("--stride", "-s", type=int, default=16, help="Stride for sliding-window inference")
    parser.add_argument("--box-size", type=int, default=48, help="Patch size for sliding-window inference")
    parser.add_argument("--apix", type=float, default=1.0, help="Override voxel size used by parse_map")
    parser.add_argument("--normalize-percentile", type=float, default=99.999, help="Positive-density percentile used for 0-1 normalization")
    parser.add_argument("--gaussian-sigma", type=float, default=None, help="Sigma for Gaussian patch fusion; defaults to box_size / 4")
    parser.add_argument(
        "--gaussian-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Gaussian-weighted patch fusion; disable with --no-gaussian-weight",
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False, help="Use autocast fp16 on CUDA for inference")
    return parser


def main(args) -> None:
    seed_torch(42)

    cpu_num = 4
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    start = time.time()
    devices = get_device_names(args.device)
    logits, origin, voxel_size, model_kwargs, density_clip_max = run_na_type_inference_on_map(
        ckpt_path=args.ckpt,
        map_file=args.input,
        apix=float(args.apix),
        box_size=int(args.box_size),
        stride=int(args.stride),
        batch_size=int(args.batchsize),
        devices=devices,
        gaussian_weight=bool(args.gaussian_weight),
        gaussian_sigma=args.gaussian_sigma,
        fp16=bool(args.fp16),
        normalize_percentile=float(args.normalize_percentile),
    )

    out_dir = Path(abspath(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "logits.npz"
    np.savez(
        out_path,
        map=logits.astype(np.float32, copy=False),
        origin=np.asarray(origin, dtype=np.float32),
        voxel_size=np.asarray(voxel_size, dtype=np.float32),
    )

    end = time.time()
    print(f"# NA-type model = {model_kwargs['module']}.{model_kwargs['class_name']}")
    print("# Class order = [A, C, G, U/T]")
    print(f"# Density clip max = {density_clip_max:.6f}")
    print(f"# Write logits to {out_path}")
    print(f"# Time consuming {end - start:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 4-class NA A/C/G/U(T) SCUNet inference and save logits.npz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = add_args(parser)
    main(parser.parse_args())
