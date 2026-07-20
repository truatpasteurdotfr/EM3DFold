import argparse
import os
import warnings

import numpy as np

from em3dfold.io.pdbio import read_pdb
from em3dfold.pipeline.eval import ALL_MOL_TYPES, _filter_structure, _print_eval_result, eval_local
from em3dfold.utils.qscore.mrc_utils import load_mrc
from em3dfold.utils.qscore.pdb_utils import get_protein_from_file_path
from em3dfold.utils.qscore.q_score import calculate_q_score

warnings.filterwarnings("ignore")


DEFAULT_QSCORE_BIN_EDGES = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 2).tolist())


def add_args(parser):
    parser.add_argument("--target", "-t", help="Target structure")
    parser.add_argument("--query", "-q", help="Query structure")
    parser.add_argument("--map", "-m", required=True, help="Density map used to compute target q-score")
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
        "--qscore-bins",
        default=",".join(str(v) for v in DEFAULT_QSCORE_BIN_EDGES),
        help="Comma-separated target-residue qscore bin edges. Example: 0,0.05,0.1,...,1.0",
    )
    parser.add_argument(
        "--min-atom-qscore", "--min-residue-qscore",
        dest="min_atom_qscore",
        type=float,
        default=-100.0,
        help="Ignore atoms with qscore below this cutoff before computing target residue qscore bins.",
    )
    return parser


def _format_edge(value):
    value = float(value)
    if np.isinf(value):
        return "inf"
    if float(value).is_integer():
        return f"{value:.1f}"
    return "{:g}".format(value)


def _parse_qscore_bin_edges(bin_text):
    if bin_text is None or not str(bin_text).strip():
        return np.asarray(DEFAULT_QSCORE_BIN_EDGES, dtype=np.float32)
    values = [float(item.strip()) for item in str(bin_text).split(",") if item.strip()]
    if not values:
        raise ValueError("qscore bin edges cannot be empty")
    edges = np.asarray(values, dtype=np.float32)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("qscore bin edges must be strictly increasing")
    return edges


def _build_bin_specs(edges):
    specs = [(f"<{_format_edge(edges[0])}", -np.inf, float(edges[0]))]
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


def _compute_target_residue_qscores(structure_path, map_path, min_atom_qscore=-100.0):
    np.random.seed(42)
    polymer = get_protein_from_file_path(structure_path)
    cryo_map = load_mrc(map_path)

    atom_mask = polymer.atom_mask.astype(bool)
    atoms = polymer.atom_positions[atom_mask]
    flat_qscores = calculate_q_score(atoms, cryo_map).astype(np.float32)

    qscore_per_atom = np.full(polymer.atom_mask.shape, np.nan, dtype=np.float32)
    qscore_per_atom[atom_mask] = flat_qscores

    valid_mask = np.logical_and(atom_mask & np.isfinite(qscore_per_atom), qscore_per_atom >= float(min_atom_qscore))
    residue_qscore = np.full((polymer.atom_mask.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid_mask):
        atom_sum = np.where(valid_mask, qscore_per_atom, 0.0).sum(axis=1, dtype=np.float64)
        atom_count = valid_mask.sum(axis=1)
        nonzero = atom_count > 0
        residue_qscore[nonzero] = (atom_sum[nonzero] / atom_count[nonzero]).astype(np.float32)
    return residue_qscore


def _evaluate_one_mol_qscore(
    mol,
    tgt_atom_pos,
    tgt_atom_mask,
    tgt_res_type,
    tgt_res_idx,
    tgt_chain_idx,
    tgt_residue_qscore,
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
    map_path = os.path.abspath(args.map) if args.map is not None else args.map

    tgt_atom_pos_m, tgt_atom_mask_m, tgt_res_type_m, tgt_res_idx_m, tgt_chain_idx_m = _filter_structure(
        tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )
    qry_atom_pos_m, qry_atom_mask_m, qry_res_type_m, qry_res_idx_m, qry_chain_idx_m = _filter_structure(
        qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
    )
    tgt_residue_qscore_m, _, _, _, _ = _filter_structure(
        tgt_residue_qscore, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )

    print("#" + "-" * 72)
    print(f"# Eval {mol}")
    print(f"# Read {len(tgt_atom_pos_m)} valid residues from {target_path}")
    print(f"# Read {len(qry_atom_pos_m)} valid residues from {query_path}")
    print(f"# Read map from {map_path}")

    if len(tgt_atom_pos_m) == 0 or len(qry_atom_pos_m) == 0:
        print(f"# Skip {mol} evaluation because target or query has no valid residues")
        return

    valid_q = np.isfinite(tgt_residue_qscore_m)
    if not np.any(valid_q):
        print(
            f"# Skip {mol} qscore-bin evaluation because no target residue has any atom q-score passing the cutoff "
            f"{float(args.min_atom_qscore):.4f}"
        )
        return

    print(
        "# Atom qscore cutoff for {} residues: qscore >= {:.4f}".format(
            mol,
            float(args.min_atom_qscore),
        )
    )
    print(
        "# Target residue qscore stats for {} residues: min = {:.4f} max = {:.4f} mean = {:.4f}".format(
            mol,
            float(np.min(tgt_residue_qscore_m[valid_q])),
            float(np.max(tgt_residue_qscore_m[valid_q])),
            float(np.mean(tgt_residue_qscore_m[valid_q])),
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
            target_mask = tgt_residue_qscore_m < upper
        elif np.isinf(upper):
            target_mask = tgt_residue_qscore_m >= lower
        else:
            target_mask = np.logical_and(tgt_residue_qscore_m >= lower, tgt_residue_qscore_m < upper)
        target_mask = np.logical_and(target_mask, valid_q)

        target_len = int(np.sum(target_mask))
        print(f"# Eval {mol} qscore bin {label} target_len = {target_len}")
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
    bin_edges = _parse_qscore_bin_edges(args.qscore_bins)
    bin_specs = _build_bin_specs(bin_edges)

    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx = read_pdb(
        args.target, return_bfactor=False
    )
    qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(
        args.query, return_bfactor=False
    )
    tgt_residue_qscore = _compute_target_residue_qscores(args.target, args.map, min_atom_qscore=args.min_atom_qscore)

    print("# Using distance cutoff = {:.4f}".format(args.d))
    print("# Target residue qscore bins = {}".format(", ".join(label for label, _, _ in bin_specs)))
    print("# Min atom qscore cutoff = {:.4f}".format(float(args.min_atom_qscore)))

    eval_mols = (args.mol,) if args.mol is not None else ALL_MOL_TYPES
    for mol in eval_mols:
        _evaluate_one_mol_qscore(
            mol,
            tgt_atom_pos,
            tgt_atom_mask,
            tgt_res_type,
            tgt_res_idx,
            tgt_chain_idx,
            tgt_residue_qscore,
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
