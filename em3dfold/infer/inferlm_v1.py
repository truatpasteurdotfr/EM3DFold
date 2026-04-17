import argparse
import os

from em3dfold.infer.inferlm_common import (
    load_model_bundle,
    prepare_common_args,
    run_main,
)
from em3dfold.utils.misc_utils import abspath
import torch


@torch.no_grad()
def run_inference_on_data_v1(
    module,
    meta_batch_list,
    run_iters: int = 2,
    seq_attention_batch_size: int = 200,
    fp16: bool = False,
    using_cache: bool = False,
):
    with_seq = ("prot_seq_embed" in meta_batch_list[0]) or ("na_seq_embed" in meta_batch_list[0])
    meta_input_list = []
    for data in meta_batch_list:
        kwargs = {
            "affines": data["affines"],
            "prot_mask": data["prot_mask"].long(),
            "run_iters": run_iters,
        }

        if data["batch_num"] == 1:
            kwargs["batch"] = None
            kwargs["cryo_grids"] = [data["cryo_grids"]]
            kwargs["cryo_global_origins"] = [data["cryo_global_origins"]]
            kwargs["cryo_voxel_sizes"] = [data["cryo_voxel_sizes"]]
            if with_seq:
                prot_seq_embed = data["prot_seq_embed"][None]
                na_seq_embed = data["na_seq_embed"][None]
                kwargs["prot_seq_embed"] = prot_seq_embed
                kwargs["prot_seq_embed_mask"] = torch.ones(1, data["prot_seq_embed"].shape[0])
                kwargs["na_seq_embed"] = na_seq_embed
                kwargs["na_seq_embed_mask"] = torch.ones(1, data["na_seq_embed"].shape[0])
        else:
            kwargs["batch"] = data["batch"]
            kwargs["cryo_grids"] = [data["cryo_grids"] for _ in range(data["batch_num"])]
            kwargs["cryo_global_origins"] = [
                data["cryo_global_origins"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_voxel_sizes"] = [
                data["cryo_voxel_sizes"] for _ in range(data["batch_num"])
            ]
            if with_seq:
                prot_seq_embed = data["prot_seq_embed"][None].expand(data["batch_num"], -1, -1)
                na_seq_embed = data["na_seq_embed"][None].expand(data["batch_num"], -1, -1)
                kwargs["prot_seq_embed"] = prot_seq_embed
                kwargs["prot_seq_embed_mask"] = torch.ones(
                    data["batch_num"], data["prot_seq_embed"].shape[0]
                )
                kwargs["na_seq_embed"] = na_seq_embed
                kwargs["na_seq_embed_mask"] = torch.ones(
                    data["batch_num"], data["na_seq_embed"].shape[0]
                )

        meta_input_list.append(kwargs)
    return module(meta_input_list)


def get_args():
    parser = prepare_common_args(argparse.ArgumentParser())
    script_dir = abspath(os.path.dirname(__file__))
    parser.add_argument(
        "--task-type",
        choices=["protein", "na"],
        default="protein",
        help="Which v1 model type to use.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to the v1 model config yaml",
    )
    return parser.parse_args()


def main(args):
    if args.model_config is None:
        config_name = (
            "model_v1_protein.yaml"
            if args.task_type == "protein"
            else "model_v1_na.yaml"
        )
        args.model_config = os.path.join(abspath(os.path.dirname(__file__)), "config", config_name)
    args.min_na_chain_len = 3 if args.task_type == "na" else 1
    model_class, model_args = load_model_bundle(args.model_config)
    model_args["model_type"] = args.task_type
    run_main(args, model_class, model_args, run_inference_on_data_v1)


if __name__ == "__main__":
    main(get_args())

