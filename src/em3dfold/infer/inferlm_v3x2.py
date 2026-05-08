import argparse
import os

import torch

from em3dfold.infer.inferlm_common import (
    load_model_bundle,
    prepare_common_args,
    run_main,
)
from em3dfold.infer.inferlm_v3 import run_inference_on_data_v3
from em3dfold.utils.misc_utils import abspath


def get_args():
    parser = prepare_common_args(argparse.ArgumentParser())
    script_dir = abspath(os.path.dirname(__file__))
    parser.add_argument(
        "--model-config",
        default=os.path.join(script_dir, "config", "model_v3x2_12l_256_128_h8.yaml"),
        help="Path to the v3x2 model config yaml",
    )
    return parser.parse_args()


def main(args):
    model_class, model_args = load_model_bundle(args.model_config)
    run_main(args, model_class, model_args, run_inference_on_data_v3)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(get_args())
