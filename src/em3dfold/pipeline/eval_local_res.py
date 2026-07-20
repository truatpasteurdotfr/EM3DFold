import argparse
import os
import warnings

import numpy as np
from scipy.ndimage import map_coordinates

from em3dfold.io.pdbio import read_pdb
from em3dfold.pipeline.eval import (
    ALL_MOL_TYPES,
    _evaluate_one_mol,
    _filter_structure,
    _print_eval_result,
    eval_local,
)
from em3dfold.utils.cryo_utils import read_map

warnings.filterwarnings("ignore")


DEFAULT_LOCAL_RES_BIN_EDGES = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def add_args(parser):
    parser.add_argument("--target", "-t", help="Target structure")
    parser.add_argument("--query", "-q", help="Query structure")
    parser.add_argument("--local-res-map", required=True, help="Local resolution map in MRC/MAP format")
    parser.add_argument("--eval-chain", action="store_true", help="Eval in chain level")
    parser.add_argument("-d", type=float, default=3.0, help="Distance cutoff")
    parser.add_argument("--align", action="store_true", default=False)
    parser.add_argument(
        "--mol",
        choices=ALL_MOL_TYPES,
        default=None,
        help="Optional compatibility selector. If omitted, evaluate protein, nucleic and complex together.",
    )
    parser.add_argument(
        "--local-res-bins",
        default=",".join(str(v) for v in DEFAULT_LOCAL_RES_BIN_EDGES),
        help="Comma-separated upper bin edges. Example: 2.5,3,3.5,4,4.5,5",
    )
    return parser


def _format_bin_value(value):
    value = float(value)
    if np.isinf(value):
        return "inf"
    return "{:g}".format(value)


def _parse_local_res_bin_edges(bin_text):
    if bin_text is None or not str(bin_text).strip():
        return np.asarray(DEFAULT_LOCAL_RES_BIN_EDGES, dtype=np.float32)
    values = [float(item.strip()) for item in str(bin_text).split(",") if item.strip()]
    if not values:
        raise ValueError("local resolution bin edges cannot be empty")
    edges = np.asarray(values, dtype=np.float32)
    if np.any(edges <= 0):
        raise ValueError("local resolution bin edges must be positive")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("local resolution bin edges must be strictly increasing")
    return edges


def _build_bin_specs(edges):
    specs = []
    lower = 0.0
    for upper in edges:
        label = f"{_format_bin_value(lower)}-{_format_bin_value(upper)}"
        specs.append((label, lower, float(upper)))
        lower = float(upper)
    specs.append((f">{_format_bin_value(edges[-1])}", float(edges[-1]), np.inf))
    return specs


def _sample_map_values(points, map_data, origin, voxel_size):
    points = np.asarray(points, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    grid_points = (points - origin[None, :]) / voxel_size[None, :]
    return map_coordinates(
        map_data,
        [grid_points[:, 2], grid_points[:, 1], grid_points[:, 0]],
        order=1,
        mode="nearest",
    )


def _print_bin_eval_result(scope_name, label, metrics, query_len, target_len):
    if metrics is None:
        print(
            f"# {scope_name} bin {label} target_len = {target_len} query_len = {query_len} "
            "CX_RMSD = - BB_RMSD = - Cov = 0.0 Seq_Match = - Seq_Recall = -"
        )
        return
    print(
        "# {} bin {} target_len = {} query_len = {} CX_RMSD = {:.4f} BB_RMSD = {:.4f} Cov = {:.4f} "
        "Seq_Match = {:.4f} Seq_Recall = {:.4f}".format(
            scope_name,
            label,
            target_len,
            query_len,
            metrics["c4_rmsd"],
            metrics["bb_rmsd"],
            metrics["cov"],
            metrics["seq_match"],
            metrics["seq_recall"],
        )
    )


def _evaluate_one_mol_local_res(
    mol,
    tgt_atom_pos,
    tgt_atom_mask,
    tgt_res_type,
    tgt_res_idx,
    tgt_chain_idx,
    qry_atom_pos,
    qry_atom_mask,
    qry_res_type,
    qry_res_idx,
    qry_chain_idx,
    local_res_map,
    local_res_origin,
    local_res_voxel_size,
    bin_specs,
    args,
):
    target_path = os.path.abspath(args.target) if args.target is not None else args.target
    query_path = os.path.abspath(args.query) if args.query is not None else args.query
    local_res_path = (
        os.path.abspath(args.local_res_map) if args.local_res_map is not None else args.local_res_map
    )

    tgt_atom_pos_m, tgt_atom_mask_m, tgt_res_type_m, tgt_res_idx_m, tgt_chain_idx_m = _filter_structure(
        tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )
    qry_atom_pos_m, qry_atom_mask_m, qry_res_type_m, qry_res_idx_m, qry_chain_idx_m = _filter_structure(
        qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
    )

    print("#" + "-" * 72)
    print(f"# Eval {mol}")
    print(f"# Read {len(tgt_atom_pos_m)} valid residues from {target_path}")
    print(f"# Read {len(qry_atom_pos_m)} valid residues from {query_path}")
    print(f"# Read local resolution map from {local_res_path}")

    if len(tgt_atom_pos_m) == 0 or len(qry_atom_pos_m) == 0:
        print(f"# Skip {mol} evaluation because target or query has no valid residues")
        return

    target_anchor_pos = tgt_atom_pos_m[..., 1, :]
    target_local_res = _sample_map_values(
        target_anchor_pos,
        local_res_map,
        local_res_origin,
        local_res_voxel_size,
    ).astype(np.float32)
    print(
        "# Local resolution stats for {} target residues: min = {:.4f} max = {:.4f} mean = {:.4f}".format(
            mol,
            float(np.min(target_local_res)),
            float(np.max(target_local_res)),
            float(np.mean(target_local_res)),
        )
    )

    print(f"# Eval {mol} all len = {len(qry_atom_pos_m)}")
    metrics = eval_local(
        tgt_atom_pos_m,
        tgt_atom_mask_m,
        tgt_res_type_m,
        qry_atom_pos_m,
        qry_atom_mask_m,
        qry_res_type_m,
        r=args.d,
        mol=mol,
        align=args.align,
    )
    _print_eval_result(f"{mol} all", metrics, len(qry_atom_pos_m))

    if args.eval_chain:
        qry_n_chain = qry_chain_idx_m.max() + 1
        for i in range(qry_n_chain):
            sel_idxs = np.arange(0, len(qry_atom_pos_m))[qry_chain_idx_m == i]
            print(f"# Eval {mol} chain {i} len = {len(sel_idxs)}")
            metrics = eval_local(
                tgt_atom_pos_m,
                tgt_atom_mask_m,
                tgt_res_type_m,
                qry_atom_pos_m[sel_idxs],
                qry_atom_mask_m[sel_idxs],
                qry_res_type_m[sel_idxs],
                r=args.d,
                mol=mol,
                align=args.align,
            )
            _print_eval_result(f"{mol} chain {i}", metrics, len(sel_idxs))

    for label, lower, upper in bin_specs:
        if np.isinf(upper):
            target_mask = target_local_res >= lower
        else:
            target_mask = np.logical_and(target_local_res >= lower, target_local_res < upper)

        target_len = int(np.sum(target_mask))
        print(f"# Eval {mol} local-res bin {label} target_len = {target_len}")
        if target_len == 0:
            _print_bin_eval_result(f"{mol} all", label, None, len(qry_atom_pos_m), 0)
            if args.eval_chain:
                qry_n_chain = qry_chain_idx_m.max() + 1
                for i in range(qry_n_chain):
                    sel_idxs = np.arange(0, len(qry_atom_pos_m))[qry_chain_idx_m == i]
                    _print_bin_eval_result(f"{mol} chain {i}", label, None, len(sel_idxs), 0)
            continue

        tgt_atom_pos_bin = tgt_atom_pos_m[target_mask]
        tgt_atom_mask_bin = tgt_atom_mask_m[target_mask]
        tgt_res_type_bin = tgt_res_type_m[target_mask]

        metrics = eval_local(
            tgt_atom_pos_bin,
            tgt_atom_mask_bin,
            tgt_res_type_bin,
            qry_atom_pos_m,
            qry_atom_mask_m,
            qry_res_type_m,
            r=args.d,
            mol=mol,
            align=args.align,
        )
        _print_bin_eval_result(f"{mol} all", label, metrics, len(qry_atom_pos_m), target_len)

        if args.eval_chain:
            qry_n_chain = qry_chain_idx_m.max() + 1
            for i in range(qry_n_chain):
                sel_idxs = np.arange(0, len(qry_atom_pos_m))[qry_chain_idx_m == i]
                metrics = eval_local(
                    tgt_atom_pos_bin,
                    tgt_atom_mask_bin,
                    tgt_res_type_bin,
                    qry_atom_pos_m[sel_idxs],
                    qry_atom_mask_m[sel_idxs],
                    qry_res_type_m[sel_idxs],
                    r=args.d,
                    mol=mol,
                    align=args.align,
                )
                _print_bin_eval_result(f"{mol} chain {i}", label, metrics, len(sel_idxs), target_len)


def main(args):
    bin_edges = _parse_local_res_bin_edges(args.local_res_bins)
    bin_specs = _build_bin_specs(bin_edges)

    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx = read_pdb(
        args.target, return_bfactor=False
    )
    qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(
        args.query, return_bfactor=False
    )
    local_res_map, local_res_origin, local_res_voxel_size = read_map(args.local_res_map, ignorestart=False)

    print("# Using distance cutoff = {:.4f}".format(args.d))
    print("# Local resolution bins = {}".format(", ".join(label for label, _, _ in bin_specs)))
    eval_mols = (args.mol,) if args.mol is not None else ALL_MOL_TYPES

    for mol in eval_mols:
        _evaluate_one_mol_local_res(
            mol,
            tgt_atom_pos,
            tgt_atom_mask,
            tgt_res_type,
            tgt_res_idx,
            tgt_chain_idx,
            qry_atom_pos,
            qry_atom_mask,
            qry_res_type,
            qry_res_idx,
            qry_chain_idx,
            local_res_map,
            local_res_origin,
            local_res_voxel_size,
            bin_specs,
            args,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
