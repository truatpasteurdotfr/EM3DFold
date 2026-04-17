import argparse
import importlib
import os
import sys
import tqdm
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf

from em3dfold.io.pdbio import chains_atom_pos_to_pdb
from em3dfold.polymer_utils.polymer import get_polymer_from_file_path
from em3dfold.utils.misc_utils import abspath, pjoin
from em3dfold.utils.torch_utils import clear_cuda_cache, seed_everything


def load_model_bundle(config_path):
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    model_cfg = cfg["model"]
    module = importlib.import_module(model_cfg["module"])
    model_class = getattr(module, model_cfg["class_name"])
    model_args = dict(model_cfg.get("args", {}))
    return model_class, model_args


def prepare_common_args(parser):
    script_dir = abspath(os.path.dirname(__file__))
    num_res_per_run = 200
    parser.add_argument("--map", "-i", required=True, help="The path to the input map")
    parser.add_argument("--polymer", "-p", required=True, help="The path to the polymer file")
    parser.add_argument(
        "--model-dir",
        "-m",
        help="Where the model weights are",
        default=pjoin(script_dir, "..", "weights", "model_all_atom"),
    )
    parser.add_argument("--output-dir", "-o", default=".", help="Where to save the results")
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument("--crop-length", type=int, default=num_res_per_run, help="How many points per batch")
    parser.add_argument("--repeat-per-residue", default=1, type=int, help="How many times to repeat per residue")
    parser.add_argument("--run-iters", default=2, type=int, help="Cycling steps for model forward")
    parser.add_argument("--batch-size", default=1, type=int, help="How many batches to run in parallel")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 in inference")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="The voxel size that the GNN should be interpolating to.",
    )
    parser.add_argument("--refine", action="store_true", help="Refine of structure")
    parser.add_argument("--no-use-random-affine", action="store_true")
    parser.add_argument("--recycle", type=int, default=3, help="Recycling times")
    parser.add_argument("--prot-seq-embed", type=str, default=None)
    parser.add_argument("--na-seq-embed", type=str, default=None)
    parser.add_argument(
        "--na-aa-logits",
        type=str,
        default=None,
        help="Path to nucleic-acid aa_logits npz predicted from segmentation network",
    )
    parser.add_argument("--protein-seq", type=str, default=None)
    parser.add_argument("--dna-seq", type=str, default=None)
    parser.add_argument("--rna-seq", type=str, default=None)
    parser.add_argument(
        "--min-na-chain-len",
        type=int,
        default=3,
        help="Minimum NA backbone-chain length kept in final outputs",
    )
    parser.set_defaults(fallback_to_predicted_na_types=True)
    parser.add_argument(
        "--no-fallback-to-predicted-na-types",
        dest="fallback_to_predicted_na_types",
        action="store_false",
        help="Do not fall back to network-predicted nucleotide types for unmatched NA chains",
    )
    parser.set_defaults(pass_prev_aa_probs=True)
    parser.add_argument(
        "--no-pass-prev-aa-probs",
        dest="pass_prev_aa_probs",
        action="store_false",
        help="Do not pass previous recycle prev_aa_probs into the next outer recycle round",
    )
    parser.set_defaults(pass_prev_rmsd=True)
    parser.add_argument(
        "--no-pass-prev-rmsd",
        dest="pass_prev_rmsd",
        action="store_false",
        help="Do not pass previous recycle prev_rmsd into the next outer recycle round",
    )
    parser.set_defaults(pass_prev_node=True)
    parser.add_argument(
        "--no-pass-prev-node",
        dest="pass_prev_node",
        action="store_false",
        help="Do not pass previous recycle prev_node into the next outer recycle round",
    )
    return parser


def _polymer_to_chain_lists(polymer):
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_type = []
    chains_res_idx = []

    unique_chain_indices = np.unique(polymer.chain_index)
    for chain_idx in unique_chain_indices:
        chain_mask = polymer.chain_index == chain_idx
        chains_atom_pos.append(polymer.atomc_positions[chain_mask])
        chains_atom_mask.append(polymer.atomc_mask[chain_mask])
        chains_res_type.append(polymer.aatype[chain_mask])
        chains_res_idx.append(polymer.residue_index[chain_mask])

    return chains_atom_pos, chains_atom_mask, chains_res_type, chains_res_idx


def _write_merged_best_so_far_output(output_path, protein_path=None, na_path=None):
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_type = []
    chains_res_idx = []

    for path in [protein_path, na_path]:
        if path is None or (not os.path.isfile(path)):
            continue
        polymer = get_polymer_from_file_path(path)
        (
            part_atom_pos,
            part_atom_mask,
            part_res_type,
            part_res_idx,
        ) = _polymer_to_chain_lists(polymer)
        chains_atom_pos.extend(part_atom_pos)
        chains_atom_mask.extend(part_atom_mask)
        chains_res_type.extend(part_res_type)
        chains_res_idx.extend(part_res_idx)

    if len(chains_atom_pos) == 0:
        return None

    chains_atom_pos_to_pdb(
        output_path,
        chains_atom_pos,
        chains_atom_mask,
        chains_res_type,
        chains_res_idx=chains_res_idx,
        suffix="cif",
    )
    return output_path


def run_inference_loop(args, model_class, model_args, run_inference_fn):
    from em3dfold.infer.denovo import final_results_align_to_sequence
    from em3dfold.infer.gnn_inference_utils import (
        argmin_random,
        collate_nn_results,
        get_final_nn_results,
        get_inference_data,
        get_neighbour_idxs,
        init_empty_collate_results,
        init_polymer_from_translation,
    )
    from em3dfold.polymer_utils.polymer import get_polymer_from_file_path
    from em3dfold.utils.cryo_utils import MRCObject, parse_map
    from em3dfold.utils.multi_gpu_wrapper import MultiGPUWrapper
    from em3dfold.utils.torch_utils import get_device_names

    os.makedirs(args.output_dir, exist_ok=True)

    device_names = get_device_names(args.device)
    num_devices = len(device_names)

    polymer = None
    if args.polymer.endswith(".cif") or args.polymer.endswith(".pdb"):
        print(f"# Read structure from {args.polymer}")
        if not args.no_use_random_affine:
            print("# Use random affine")
            polymer = init_polymer_from_translation(args.polymer)
        else:
            print("# Use affine from file")
            polymer = get_polymer_from_file_path(args.polymer)

        prot_seq_dim = model_args.get("d_seq", 1280)
        na_seq_dim = model_args.get("d_seq_na", prot_seq_dim)
        if args.prot_seq_embed is not None:
            prot_seq_embed = np.load(args.prot_seq_embed)
        else:
            prot_seq_embed = np.zeros((3, prot_seq_dim), dtype=np.float32)
        if args.na_seq_embed is not None:
            na_seq_embed = np.load(args.na_seq_embed)
        else:
            na_seq_embed = np.zeros((3, na_seq_dim), dtype=np.float32)
        polymer.prot_seq_embedding = prot_seq_embed.astype(np.float32)
        polymer.na_seq_embedding = na_seq_embed.astype(np.float32)
        print("# Load prot seq embed len = {}".format(polymer.prot_seq_embedding.shape))
        print("# Load na   seq embed len = {}".format(polymer.na_seq_embedding.shape))

        prev_recycle_state = (
            getattr(args, "prev_recycle_state", None)
            if (
                getattr(args, "pass_prev_aa_probs", True)
                or getattr(args, "pass_prev_rmsd", True)
                or getattr(args, "pass_prev_node", True)
            )
            else None
        )
        loaded_prev_aa = False
        loaded_prev_rmsd = False
        loaded_prev_node = False
        if prev_recycle_state is not None:
            if getattr(args, "pass_prev_aa_probs", True):
                prev_aa_probs = prev_recycle_state.get("prev_aa_probs")
                if prev_aa_probs is not None:
                    prev_aa_probs = np.asarray(prev_aa_probs, dtype=np.float32)
                    if len(prev_aa_probs) != len(polymer.rigidgroups_gt_frames):
                        warnings.warn(
                            "Skip previous recycle aa probabilities because residue count does not match "
                            f"({len(prev_aa_probs)} vs {len(polymer.rigidgroups_gt_frames)})."
                        )
                    else:
                        polymer.prev_aa_probs = prev_aa_probs
                        loaded_prev_aa = True
                        print("# Load prev recycle aa probs shape = {}".format(prev_aa_probs.shape))

            if getattr(args, "pass_prev_rmsd", True):
                prev_rmsd = prev_recycle_state.get("prev_rmsd")
                if prev_rmsd is not None:
                    prev_rmsd = np.asarray(prev_rmsd, dtype=np.float32).reshape(-1, 1)
                    if len(prev_rmsd) != len(polymer.rigidgroups_gt_frames):
                        warnings.warn(
                            "Skip previous recycle rmsd because residue count does not match "
                            f"({len(prev_rmsd)} vs {len(polymer.rigidgroups_gt_frames)})."
                        )
                    else:
                        polymer.prev_rmsd = prev_rmsd
                        loaded_prev_rmsd = True
                        print("# Load prev recycle rmsd shape = {}".format(prev_rmsd.shape))

            if getattr(args, "pass_prev_node", True):
                prev_node = prev_recycle_state.get("prev_node")
                if prev_node is not None:
                    prev_node = np.asarray(prev_node, dtype=np.float32)
                    if len(prev_node) != len(polymer.rigidgroups_gt_frames):
                        warnings.warn(
                            "Skip previous recycle node states because residue count does not match "
                            f"({len(prev_node)} vs {len(polymer.rigidgroups_gt_frames)})."
                        )
                    else:
                        polymer.prev_node = prev_node
                        loaded_prev_node = True
                        print("# Load prev recycle node shape = {}".format(prev_node.shape))
        print(
            "# Prev recycle state loaded: pass_prev_aa_probs={} pass_prev_rmsd={} pass_prev_node={} aa_probs={} rmsd={} node={}".format(
                getattr(args, "pass_prev_aa_probs", True),
                getattr(args, "pass_prev_rmsd", True),
                getattr(args, "pass_prev_node", True),
                loaded_prev_aa,
                loaded_prev_rmsd,
                loaded_prev_node,
            )
        )

    if polymer is None:
        raise RuntimeError(f"File {args.polymer} is not a supported file format.")

    if not (args.map.endswith(".mrc") or args.map.endswith(".map")):
        warnings.warn(
            f"The file {args.map} does not end with '.mrc' or '.map'\nPlease make sure it is an MRC file."
        )

    grid, origin, _, voxel_size = parse_map(
        args.map, False, args.voxel_size, device=device_names[0]
    )
    maximum = np.percentile(grid[grid > 0.0], 99.999)
    grid = np.clip(grid, a_min=0.0, a_max=maximum)
    grid = grid / (maximum + 1e-6)
    grid = (grid - grid.mean()) / (grid.std() + 1e-6)
    print(f"# Grid mean: {grid.mean():.6f}")
    print(f"# Grid min:  {grid.min():.6f}")
    print(f"# Grid max:  {grid.max():.6f}")

    grid_data = MRCObject(grid=grid, origin=origin, voxel_size=voxel_size)

    na_aa_logits_data = None
    if getattr(args, "na_aa_logits", None) is not None:
        npz = np.load(args.na_aa_logits)
        na_aa_logits_data = {
            "map": np.asarray(npz["map"], dtype=np.float32),
            "origin": np.asarray(npz["origin"], dtype=np.float32),
            "voxel_size": np.asarray(npz["voxel_size"], dtype=np.float32),
        }
        print(
            "# Load NA aa logits map shape = {}".format(
                na_aa_logits_data["map"].shape
            )
        )

    num_res = len(polymer.rigidgroups_gt_frames)
    collated_results = init_empty_collate_results(num_res, device="cpu")
    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    print(f"# Run iters = {args.run_iters}")
    print(f"# Repeat per res = {args.repeat_per_residue}")

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)
    crop_length_for_infer = min(args.crop_length, num_res)
    num_pred_residues = (
        max(crop_length_for_infer // 2, 1)
        if num_res > args.crop_length
        else num_res
    )
    init_neighbours = get_neighbour_idxs(polymer, k=num_pred_residues)

    with MultiGPUWrapper(
        model_class,
        model_args,
        args.model_dir,
        device_names,
        args.fp16,
    ) as wrapper:
        while residues_left > 0:
            idxs = argmin_random(
                collated_results["counts"],
                init_neighbours,
                args.batch_size * num_devices,
                args.repeat_per_residue,
            )
            data = get_inference_data(
                polymer,
                grid_data,
                idxs,
                crop_length=args.crop_length,
                num_devices=num_devices,
                na_aa_logits_data=na_aa_logits_data,
            )
            run_iters = args.run_iters if args.run_iters >= 1 else 1
            results = run_inference_fn(
                wrapper,
                data,
                fp16=args.fp16,
                run_iters=run_iters,
            )
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
                - torch.sum(
                    collated_results["counts"] > args.repeat_per_residue - 1
                ).item()
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
    print("# Get all-atom position")
    postprocess_na_aa_logits_data = (
        na_aa_logits_data if getattr(args, "prune", False) else None
    )
    print(
        "# NA aa logits fallback in postprocess: enabled={}".format(
            postprocess_na_aa_logits_data is not None
        )
    )
    output_info = final_results_align_to_sequence(
        final_results,
        args.protein_seq,
        args.dna_seq,
        args.rna_seq,
        args.output_dir,
        flag_prune_and_connect_chains=args.prune,
        protein_radius_threshold=args.protein_radius_threshold,
        na_radius_threshold=args.na_radius_threshold,
        min_na_chain_len=getattr(args, "min_na_chain_len", 1),
        fallback_to_predicted_na_types=getattr(args, "fallback_to_predicted_na_types", True),
        na_aa_logits_data=postprocess_na_aa_logits_data,
    )
    clear_cuda_cache(args.device, note="inferlm postprocess")
    print("# Done")
    return output_info


def run_main(args, model_class, model_args, run_inference_fn):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(42)

    if not hasattr(args, "prev_recycle_state"):
        args.prev_recycle_state = None

    output_dir = args.output_dir
    n_round_refine = max(args.recycle, 1)
    best_so_far_protein_path = None
    best_so_far_protein_num_res = -1
    last_na_after_prune_path = None

    for i in range(n_round_refine):
        args.no_use_random_affine = i != 0
        if i == n_round_refine - 1:
            args.prune = True
            args.protein_radius_threshold = 1.50
            args.na_radius_threshold = 2.50
        else:
            args.prune = False
            args.protein_radius_threshold = 1.00
            args.na_radius_threshold = 2.00

        print(f"# Infer {i + 1} / {n_round_refine}")
        args.output_dir = os.path.join(output_dir, f"recycle_{i}")
        output_info = run_inference_loop(args, model_class, model_args, run_inference_fn)
        clear_cuda_cache(args.device, note=f"inferlm recycle {i + 1}")

        protein_after_path = output_info.get("protein_after_prune_path")
        protein_after_num_res = int(output_info.get("protein_after_num_res", 0) or 0)
        if i < 2 and protein_after_path is not None:
            print(
                "# Skip best_so_far_protein update at recycle_{}; only consider rounds with index >= 2".format(
                    i
                )
            )
        if i >= 2 and protein_after_path is not None and protein_after_num_res > best_so_far_protein_num_res:
            best_so_far_protein_path = protein_after_path
            best_so_far_protein_num_res = protein_after_num_res
            print(
                "# Update best_so_far_protein: residues={} path={}".format(
                    best_so_far_protein_num_res,
                    best_so_far_protein_path,
                )
            )

        last_na_after_prune_path = output_info.get("na_after_prune_path")
        args.polymer = output_info.get("before_prune_path") or output_info.get("output_path")
        if (
            getattr(args, "pass_prev_aa_probs", True)
            or getattr(args, "pass_prev_rmsd", True)
            or getattr(args, "pass_prev_node", True)
        ):
            args.prev_recycle_state = output_info.get("next_round_state")
        else:
            args.prev_recycle_state = None
        next_round_state = args.prev_recycle_state or {}
        next_prev_aa = next_round_state.get("prev_aa_probs")
        next_prev_rmsd = next_round_state.get("prev_rmsd")
        next_prev_node = next_round_state.get("prev_node")
        print(
            "# Next recycle state prepared: pass_prev_aa_probs={} pass_prev_rmsd={} pass_prev_node={} aa_probs={} rmsd={} node={}".format(
                getattr(args, "pass_prev_aa_probs", True),
                getattr(args, "pass_prev_rmsd", True),
                getattr(args, "pass_prev_node", True),
                next_prev_aa is not None,
                next_prev_rmsd is not None,
                next_prev_node is not None,
            )
        )

    final_output_path = os.path.join(output_dir, "output.cif")
    merged_output_path = _write_merged_best_so_far_output(
        final_output_path,
        protein_path=best_so_far_protein_path,
        na_path=last_na_after_prune_path,
    )
    if merged_output_path is not None:
        print(f"# Final merged output written to {merged_output_path}")
        if best_so_far_protein_path is not None:
            print(
                "# Final protein source = {} ({} residues)".format(
                    best_so_far_protein_path,
                    best_so_far_protein_num_res,
                )
            )
        if last_na_after_prune_path is not None:
            print(f"# Final NA source = {last_na_after_prune_path}")

    print("# Done all rounds")

