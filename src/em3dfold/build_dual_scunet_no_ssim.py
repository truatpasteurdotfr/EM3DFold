from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from em3dfold import build_dual
from em3dfold.pipeline import pred_dual
from em3dfold.utils.cryo_utils import parse_map, write_map


def _postprocess_c4_map(c4_path: str) -> None:
    c4_map, origin, _, voxel_size = parse_map(c4_path, ignorestart=False, apix=1.0)
    c4_map = np.clip(c4_map, a_min=0.0, a_max=None).astype(np.float32, copy=False)
    max_value = float(np.max(c4_map)) if c4_map.size > 0 else 0.0
    if max_value > 0.0:
        c4_map = (c4_map / max_value) * 100.0
    else:
        c4_map = np.zeros_like(c4_map, dtype=np.float32)
    write_map(c4_path, c4_map.astype(np.float32, copy=False), voxel_size, origin=origin)


def _patch_pred_dual_save_outputs():
    original_save_dual_outputs = pred_dual.save_dual_outputs

    def patched_save_dual_outputs(*args, **kwargs):
        original_save_dual_outputs(*args, **kwargs)
        out_dir = args[0] if args else kwargs["out_dir"]
        c4_path = Path(out_dir) / "c4.mrc"
        if c4_path.exists():
            _postprocess_c4_map(str(c4_path))

    pred_dual.save_dual_outputs = patched_save_dual_outputs
    return original_save_dual_outputs


def _resolve_base_weights_root(args) -> Path:
    script_dir = str(Path(build_dual.__file__).resolve().parent)
    return Path(build_dual._resolve_pred_weights_dir(args.pred_weights_dir, script_dir)).resolve()


def _patch_dual_weights_path(base_root: Path, dual_weights_path: Path):
    original_pjoin = build_dual.pjoin
    base_root_str = str(base_root)
    dual_weights_str = str(dual_weights_path)

    def patched_pjoin(*parts):
        if len(parts) == 3 and str(parts[0]) == base_root_str and parts[1] == "cpx" and parts[2] == "model_dual":
            return dual_weights_str
        return original_pjoin(*parts)

    build_dual.pjoin = patched_pjoin
    return original_pjoin


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    build_dual.add_args(parser)
    parser.add_argument(
        "--scunet-no-ssim-weights",
        required=True,
        help="Path to the SCUNet-no-SSIM dual-head stage1 weights used instead of <weights>/cpx/model_dual.",
    )
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EM3DFold build entry using a user-specified SCUNet-no-SSIM dual-head stage1 model."
    )
    add_args(parser)
    args = parser.parse_args(argv)

    dual_weights_path = Path(args.scunet_no_ssim_weights).expanduser().resolve()
    if not dual_weights_path.exists():
        raise FileNotFoundError(f"SCUNet-no-SSIM weights not found: {dual_weights_path}")

    base_root = _resolve_base_weights_root(args)
    original_save_dual_outputs = _patch_pred_dual_save_outputs()
    original_pjoin = _patch_dual_weights_path(base_root, dual_weights_path)

    try:
        return build_dual.main(args)
    finally:
        pred_dual.save_dual_outputs = original_save_dual_outputs
        build_dual.pjoin = original_pjoin


if __name__ == "__main__":
    raise SystemExit(main())
