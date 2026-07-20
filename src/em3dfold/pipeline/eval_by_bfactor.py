import argparse
import os
import warnings

import numpy as np

from em3dfold.io.pdbio import read_pdb
from em3dfold.pipeline.eval import (
    ALL_MOL_TYPES,
    _filter_structure,
    _print_eval_result,
    eval_local,
)

warnings.filterwarnings("ignore")


DEFAULT_BFACTOR_BIN_EDGES = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def add_args(parser):
    parser.add_argument("--target", "-t", help="Target structure")
    parser.add_argument("--query", "-q", help="Query structure")
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
        "--bfactor-bins",
        default=",".join(str(v) for v in DEFAULT_BFACTOR_BIN_EDGES),
        help="Comma-separated normalized-bfactor bin edges. Example: -2,-1,0,1,2",
    )
    parser.add_argument(
        "--bfactor-atom-index",
        type=int,
        default=1,
        help="Atom index used for residue-level bfactor. Default 1 matches the eval anchor atom.",
    )
    return parser


def _format_edge(value):
    value = float(value)
    if np.isinf(value):
        return "inf"
    if float(value).is_integer():
        return str(int(value))
    return "{:g}".format(value)


def _parse_bfactor_bin_edges(bin_text):
    if bin_text is None or not str(bin_text).strip():
        return np.asarray(DEFAULT_BFACTOR_BIN_EDGES, dtype=np.float32)
    values = [float(item.strip()) for item in str(bin_text).split(",") if item.strip()]
    if not values:
        raise ValueError("bfactor bin edges cannot be empty")
    edges = np.asarray(values, dtype=np.float32)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bfactor bin edges must be strictly increasing")
    return edges


def _build_bin_specs(edges):
    specs = []
    specs.append((f"<{_format_edge(edges[0])}", -np.inf, float(edges[0])))
    for lower, upper in zip(edges[:-1], edges[1:]):
        specs.append((f"{_format_edge(lower)}-{_format_edge(upper)}", float(lower), float(upper)))
    specs.append((f">{_format_edge(edges[-1])}", float(edges[-1]), np.inf))
    return specs


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


def _get_residue_bfactor_zscore(atom_mask, bfactor, atom_index):
    atom_mask = np.asarray(atom_mask)
    bfactor = np.asarray(bfactor, dtype=np.float32)
    if len(atom_mask) == 0:
        return np.empty((0,), dtype=np.float32)
    if atom_index < 0 or atom_index >= atom_mask.shape[1]:
        raise ValueError(f"bfactor atom index {atom_index} is out of range for atom dimension {atom_mask.shape[1]}")

    residue_b = bfactor[:, atom_index].astype(np.float32)
    valid = atom_mask[:, atom_index].astype(bool) & np.isfinite(residue_b)
    if not np.any(valid):
        return np.full((len(atom_mask),), np.nan, dtype=np.float32)

    mean = float(np.mean(residue_b[valid]))
    std = float(np.std(residue_b[valid]))
    if std < 1e-8:
        std = 1.0
    zscore = (residue_b - mean) / std
    zscore[~valid] = np.nan
    return zscore.astype(np.float32)


def _evaluate_one_mol_bfactor(
    mol,
    tgt_atom_pos,
    tgt_atom_mask,
    tgt_res_type,
    tgt_res_idx,
    tgt_chain_idx,
    tgt_bfactor,
    qry_atom_pos,
    qry_atom_mask,
    qry_res_type,
    qry_res_idx,
    qry_chain_idx,
    bin_specs,
    args,
):
    target_path = os.path.abspath(args.target) if args.target is not None else args.target
    query_path = os.path.abspath(args.query) if args.query is not None else args.query

    tgt_atom_pos_m, tgt_atom_mask_m, tgt_res_type_m, tgt_res_idx_m, tgt_chain_idx_m = _filter_structure(
        tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )
    qry_atom_pos_m, qry_atom_mask_m, qry_res_type_m, qry_res_idx_m, qry_chain_idx_m = _filter_structure(
        qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
    )
    tgt_bfactor_m, _, _, _, _ = _filter_structure(
        tgt_bfactor, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )

    print("#" + "-" * 72)
    print(f"# Eval {mol}")
    print(f"# Read {len(tgt_atom_pos_m)} valid residues from {target_path}")
    print(f"# Read {len(qry_atom_pos_m)} valid residues from {query_path}")

    if len(tgt_atom_pos_m) == 0 or len(qry_atom_pos_m) == 0:
        print(f"# Skip {mol} evaluation because target or query has no valid residues")
        return

    target_bfactor_z = _get_residue_bfactor_zscore(
        tgt_atom_mask_m,
        tgt_bfactor_m,
        args.bfactor_atom_index,
    )
    valid_z = np.isfinite(target_bfactor_z)
    if not np.any(valid_z):
        print(f"# Skip {mol} bfactor-bin evaluation because no valid target bfactor is available")
        return

    print(
        "# Normalized target bfactor stats for {} residues: min = {:.4f} max = {:.4f} mean = {:.4f} std = {:.4f}".format(
            mol,
            float(np.min(target_bfactor_z[valid_z])),
            float(np.max(target_bfactor_z[valid_z])),
            float(np.mean(target_bfactor_z[valid_z])),
            float(np.std(target_bfactor_z[valid_z])),
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
        if np.isneginf(lower):
            target_mask = target_bfactor_z < upper
        elif np.isinf(upper):
            target_mask = target_bfactor_z >= lower
        else:
            target_mask = np.logical_and(target_bfactor_z >= lower, target_bfactor_z < upper)
        target_mask = np.logical_and(target_mask, valid_z)

        target_len = int(np.sum(target_mask))
        print(f"# Eval {mol} bfactor bin {label} target_len = {target_len}")
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
    bin_edges = _parse_bfactor_bin_edges(args.bfactor_bins)
    bin_specs = _build_bin_specs(bin_edges)

    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, tgt_bfactor = read_pdb(
        args.target, return_bfactor=True
    )
    qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(
        args.query, return_bfactor=False
    )

    print("# Using distance cutoff = {:.4f}".format(args.d))
    print("# Bfactor atom index = {}".format(args.bfactor_atom_index))
    print("# Normalized bfactor bins = {}".format(", ".join(label for label, _, _ in bin_specs)))

    eval_mols = (args.mol,) if args.mol is not None else ALL_MOL_TYPES
    for mol in eval_mols:
        _evaluate_one_mol_bfactor(
            mol,
            tgt_atom_pos,
            tgt_atom_mask,
            tgt_res_type,
            tgt_res_idx,
            tgt_chain_idx,
            tgt_bfactor,
            qry_atom_pos,
            qry_atom_mask,
            qry_res_type,
            qry_res_idx,
            qry_chain_idx,
            bin_specs,
            args,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
