"""NA-SS variant of build_dual.

Protein processing is identical to em3dfold.build_dual. The only default
changes are for the nucleic-acid denovo branch:
1. use the v3x2_na_ss model config by default;
2. read NA all-atom weights from <weights>/na/model_all_atom_na_ss by default.
"""
import argparse
import os

from em3dfold import build_dual
from em3dfold.utils.misc_utils import pjoin


def add_args(parser):
    return build_dual.add_args(parser)


def _apply_na_ss_defaults(args):
    script_dir = os.path.dirname(build_dual.__file__)
    if getattr(args, "na_model_config", None) is None:
        args.na_model_config = pjoin(
            script_dir,
            "infer",
            "config",
            "model_v3x2_na_ss.yaml",
        )
    if getattr(args, "na_all_atom_weights", None) is None:
        weights_root_dir = build_dual._resolve_pred_weights_dir(args.pred_weights_dir, script_dir)
        args.na_all_atom_weights = pjoin(weights_root_dir, "na", "model_all_atom_na_ss")
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run EM3DFold dual build with the standard protein branch and the "
            "NA-SS nucleic-acid denovo branch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args(parser)
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    args = _apply_na_ss_defaults(args)
    print(f"# build_dual_na_ss: default NA config = {args.na_model_config}")
    print(f"# build_dual_na_ss: default NA weights = {args.na_all_atom_weights}")
    return build_dual.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
