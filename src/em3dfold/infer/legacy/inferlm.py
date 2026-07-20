import os
import sys
import tqdm
import torch
import shutil
import tempfile
import warnings
import argparse
import numpy as np
from collections import namedtuple

from em3dfold.models.v2.model_refresh import Model

from em3dfold.polymer_utils import residue_constants as rc
from em3dfold.polymer_utils.polymer import get_polymer_from_file_path, add_lm_embeddings_to_polymer

from em3dfold.utils.cryo_utils import parse_map, write_map, enlarge_grid, MRCObject
from em3dfold.utils.misc_utils import pjoin, abspath
from em3dfold.utils.torch_utils import seed_everything, get_device_names
from em3dfold.utils.multi_gpu_wrapper import MultiGPUWrapper


from em3dfold.infer.gnn_inference_utils import (
    init_empty_collate_results,
    init_polymer_from_translation,
    get_neighbour_idxs,
    argmin_random,
    get_inference_data,
    run_inference_on_data,
    collate_nn_results,
    get_final_nn_results,
)

from em3dfold.infer.denovo import final_results_align_to_sequence

def infer(args):
    # Set output dir
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.dirname(args.output_dir)

    # Device
    device_names = get_device_names(args.device)
    num_devices = len(device_names)

    # Read polymer
    polymer = None
    if args.polymer.endswith(".cif") or args.polymer.endswith(".pdb"):
        print("# Read structure from {}".format(args.polymer))
        if not args.no_use_random_affine:
            print("# Use random affine")
            polymer = init_polymer_from_translation(args.polymer)
        else:
            print("# Use affine from file")
            polymer = get_polymer_from_file_path(args.polymer)

        # Add seq embed to polymer
        if args.seq_embed is not None:
            seq_embed = np.load(args.seq_embed)
        else:
            seq_embed = np.zeros((3, 1280), dtype=np.float32)

        polymer = add_lm_embeddings_to_polymer(polymer, seq_embed)
        print("# Load seq embed")

    if polymer is None:
        raise RuntimeError(f"File {args.polymer} is not a supported file format.")

    if not (args.map.endswith(".mrc") or args.map.endswith(".map")):
        warnings.warn(f"The file {args.map} does not end with '.mrc' or '.map'\nPlease make sure it is an MRC file.")

    # Read map data and process grid (this does not change grid origin)
    grid, origin, _, voxel_size = parse_map(args.map, False, args.voxel_size, device=device_names[0])
    maximum = np.percentile(grid[grid > 0.0], 99.999)
    grid = np.clip(grid, a_min=0.0, a_max=maximum)
    grid = grid / (maximum + 1e-6)

    # STD norm
    grid = (grid - grid.mean()) / (grid.std() + 1e-6)
    print("# Grid mean: {:.6f}".format(grid.mean()))
    print("# Grid min:  {:.6f}".format(grid.min()))
    print("# Grid max:  {:.6f}".format(grid.max()))

    # Get grid
    grid_data = MRCObject(
        grid=grid,
        origin=origin,
        voxel_size=voxel_size,
    )

    # Process data
    num_res = len(polymer.rigidgroups_gt_frames)

    collated_results = init_empty_collate_results(num_res, device="cpu",)

    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    print("# Run iters = {}".format(args.run_iters))
    print("# Repeat per res = {}".format(args.repeat_per_residue))

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)

    # Get an initial set of pointers to neighbours for more efficient inference
    crop_length_for_infer = min(args.crop_length, num_res)
    num_pred_residues = (
        max(crop_length_for_infer // 2, 1)
        if num_res > args.crop_length
        else num_res
    )
    init_neighbours = get_neighbour_idxs(polymer, k=num_pred_residues)

    model_class = Model
    model_args = {
        'n_block': 12, # 12
        'use_checkpoint': False,
        'd_node': 256,
        'd_edge': 128,
        'd_head': 48,
        'n_qk_point': 4,
        'n_v_point': 8,
        'n_head': 8,
        'k': 32,
        'c_grid': 1,
        'pred_node_exist': False, 
        'pred_edge_exist': True, 
        'pred_pairing': False,
    }
    state_dict_path = args.model_dir

    with MultiGPUWrapper(model_class, model_args, state_dict_path, device_names, args.fp16) as wrapper:
        while residues_left > 0:
            idxs = argmin_random(
                collated_results["counts"], 
                init_neighbours, 
                args.batch_size * num_devices,
                args.repeat_per_residue, 
            )
            data = get_inference_data(
                polymer, grid_data, idxs, crop_length=args.crop_length, num_devices=num_devices,
            )
            run_iters = args.run_iters if args.run_iters >= 1 else 1
            results = run_inference_on_data(wrapper, data, fp16=args.fp16, run_iters=run_iters)
            for device_id in range(num_devices):
                for i in range(args.batch_size):
                    collated_results, polymer = collate_nn_results(
                        collated_results,
                        results[device_id],
                        data[device_id]["indices"],
                        polymer,
                        offset=i * args.crop_length,
                        num_pred_residues=num_pred_residues,
                    )
            residues_left = (
                num_res
                - torch.sum(collated_results["counts"] > args.repeat_per_residue - 1).item()
            )
            steps_left = (
                total_steps
                - torch.sum(
                    collated_results["counts"].clip(0, args.repeat_per_residue)
                ).item()
            )
            pbar.update(n=int(steps_left_last - steps_left))
            steps_left_last = steps_left

    pbar.close()

    final_results = get_final_nn_results(collated_results)

    if "cuda" in device_names[0]:
        print("# Clean CUDA cache")
        #torch.cuda.empty_cache()


    #############################
    ### Get all-atom position ###
    #############################
    print("# Get all-atom position")
    output_info = final_results_align_to_sequence(
        final_results,
        args.protein_seq,
        args.dna_seq,
        args.rna_seq,
        args.output_dir,
        flag_prune_and_connect_chains=args.prune,
        ca_radius_threshold=args.ca_radius_threshold,
    )
    print("# Done")
    return output_info

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # seed everything
    seed_everything(42)

    # Set final output dir
    output_dir = args.output_dir

    # Infer rounds
    n_round_refine = args.recycle
    if n_round_refine < 1:
        n_round_refine = 1

    last_output_dir = None
    for i in range(n_round_refine):
        if i == 0:
            args.no_use_random_affine = False
        else:
            args.no_use_random_affine = True

        if i == n_round_refine - 1:
            args.prune = True
            args.ca_radius_threshold = 1.50
        else:
            args.prune = False
            args.ca_radius_threshold = 0.50

        print("# Infer {} / {}".format(i + 1, n_round_refine))

        # Set output dir
        args.output_dir = os.path.join(output_dir, f"recycle_{i}")

        output_info = infer(args)

        last_output_dir = output_info.get("before_prune_path") or output_info.get("output_path")
        args.polymer = last_output_dir

    print("# Done all rounds")

def get_args():
    script_dir = abspath(os.path.dirname(__file__))
    num_res_per_run = 200
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "-i", required=True, help="The path to the input map")
    parser.add_argument(
        "--polymer", "-p", required=True, help="The path to the polymer file"
    )
    parser.add_argument(
        "--model-dir",
        "-m",
        help="Where the model at",
        default=pjoin(script_dir, "..", "weights", "model_all_atom"),
    )
    parser.add_argument("--output-dir", "-o", default=".", help="Where to save the results")
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument(
        "--crop-length", type=int, default=num_res_per_run, help="How many points per batch"
    )
    parser.add_argument(
        "--repeat-per-residue",
        default=1,
        type=int,
        help="How many times to repeat per residue",
    )
    parser.add_argument(
        "--run-iters",
        default=2,
        type=int,
        help="Cycling steps for model forward",
    )
    parser.add_argument(
        "--batch-size", default=1, type=int, help="How many batches to run in parallel"
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16 in inference")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="The voxel size that the GNN should be interpolating to."
    )
    # For refine only
    parser.add_argument(
        "--refine",
        action='store_true',
        help="Refine of structure",
    )

    parser.add_argument(
        "--no-use-random-affine",
        action='store_true',
    )
    parser.add_argument(
        "--recycle",
        type=int,
        default=3,
        help="Recycling times"
    )
    parser.add_argument(
        "--seq-embed",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--protein-seq",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dna-seq",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--rna-seq",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)

