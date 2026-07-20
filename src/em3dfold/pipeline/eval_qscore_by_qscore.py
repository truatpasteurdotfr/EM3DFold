import argparse
import warnings

import numpy as np
from scipy.spatial import KDTree

from em3dfold.io.pdbio import read_pdb
from em3dfold.pipeline.eval import ALL_MOL_TYPES, _filter_structure, _normalize_na_restypes
from em3dfold.utils.geometry import kabsch
from em3dfold.utils.qscore.mrc_utils import load_mrc
from em3dfold.utils.qscore.pdb_utils import get_protein_from_file_path
from em3dfold.utils.qscore.q_score import calculate_q_score

warnings.filterwarnings("ignore")


def add_args(parser):
    parser.add_argument("--target", "-t", required=True, help="Target structure")
    parser.add_argument("--query", "-q", required=True, help="Query structure")
    parser.add_argument("--map", "-m", required=True, help="Density map used to compute q-score")
    parser.add_argument("-d", type=float, default=3.0, help="Distance cutoff")
    parser.add_argument("--align", action="store_true", default=False)
    parser.add_argument(
        "--mol",
        choices=ALL_MOL_TYPES,
        default=None,
        help="Residue scope used for eval correspondence and q-score reporting. If omitted, print nucleic/protein/complex.",
    )
    return parser


def _compute_residue_qscores(structure_path):
    np.random.seed(42)
    polymer = get_protein_from_file_path(structure_path)
    atom_mask = polymer.atom_mask.astype(bool)
    return polymer, atom_mask


def _calculate_residue_qscores(polymer, atom_mask, cryo_map):
    atoms = polymer.atom_positions[atom_mask]
    flat_qscores = calculate_q_score(atoms, cryo_map).astype(np.float32)
    atom_qscores = np.full(polymer.atom_mask.shape, np.nan, dtype=np.float32)
    atom_qscores[atom_mask] = flat_qscores
    valid_mask = atom_mask & np.isfinite(atom_qscores)
    residue_qscores = np.full((polymer.atom_mask.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid_mask):
        atom_sum = np.where(valid_mask, atom_qscores, 0.0).sum(axis=1, dtype=np.float64)
        atom_count = valid_mask.sum(axis=1)
        nonzero = atom_count > 0
        residue_qscores[nonzero] = (atom_sum[nonzero] / atom_count[nonzero]).astype(np.float32)
    return residue_qscores


def _build_correspondence(
    target_atom_pos,
    target_atom_mask,
    target_res_type,
    query_atom_pos,
    query_atom_mask,
    query_res_type,
    r=3.0,
    align=False,
):
    del target_atom_mask, query_atom_mask
    if len(target_atom_pos) == 0 or len(query_atom_pos) == 0:
        return None, None

    target_anchor_pos = target_atom_pos[..., 1, :]
    query_anchor_pos = query_atom_pos[..., 1, :]
    tree = KDTree(target_anchor_pos)
    idxs = tree.query_ball_point(query_anchor_pos, r=r + 1e-3)

    sorted_idxs = []
    for q_point, target_idxs in zip(query_anchor_pos, idxs):
        target_idxs = np.asarray(target_idxs, dtype=np.int32)
        if len(target_idxs) == 0:
            sorted_idxs.append([])
            continue
        dists = np.linalg.norm(target_anchor_pos[target_idxs] - q_point, axis=1)
        order = np.argsort(dists)
        sorted_idxs.append(np.asarray(target_idxs)[order])

    correspondence = []
    target_idx_used = set()
    matched_query_mask = np.zeros((len(query_atom_pos),), dtype=bool)
    matched_target_mask = np.zeros((len(target_atom_pos),), dtype=bool)
    for query_idx, target_idxs in enumerate(sorted_idxs):
        unused_idx = None
        for target_idx in target_idxs:
            if target_idx in target_idx_used:
                continue
            unused_idx = target_idx
            break
        if unused_idx is not None:
            target_idx_used.add(unused_idx)
            correspondence.append([query_idx, unused_idx])
            matched_query_mask[query_idx] = True
            matched_target_mask[unused_idx] = True

    correspondence = np.asarray(correspondence, dtype=np.int32)
    if len(correspondence) == 0:
        return None, matched_query_mask

    query_atom_pos_eval = np.asarray(query_atom_pos, dtype=np.float32).copy()
    if align:
        R, t = kabsch(
            query_atom_pos_eval[correspondence[:, 0], 1, :],
            target_anchor_pos[correspondence[:, 1]],
        )
        query_atom_pos_eval = np.einsum("ij,lnj->lni", R, query_atom_pos_eval) + t
    seq_correct_mask = target_res_type[correspondence[:, 1]] == query_res_type[correspondence[:, 0]]
    return (correspondence, seq_correct_mask, matched_query_mask, matched_target_mask)


def _filter_qscore_by_mol(residue_qscore, atom_mask, res_type, res_idx, chain_idx, mol):
    del res_idx, chain_idx
    if mol == "protein":
        keep_mask = np.logical_and(atom_mask[..., 1].astype(bool), res_type < 20)
    elif mol == "nucleic":
        keep_mask = np.logical_and(atom_mask[..., 1].astype(bool), res_type >= 20)
    else:
        keep_mask = atom_mask[..., 1].astype(bool)
    return residue_qscore[keep_mask], _normalize_na_restypes(res_type[keep_mask])


def _format_float(value):
    if not np.isfinite(value):
        return "-"
    return f"{float(value):.6f}"


def _print_group_rows(label_prefix, rows):
    print(f"# {label_prefix} count = {len(rows)}")
    print(f"# {label_prefix} target_chain  target_resid  target_qscore  query_chain  query_resid  query_qscore")
    for row in rows:
        print("# {}  {}  {}  {}  {}  {}  {}".format(label_prefix, *row))


def _run_one_mol(
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
    qry_residue_qscore,
    args,
):
    tgt_atom_pos_m, tgt_atom_mask_m, tgt_res_type_m, tgt_res_idx_m, tgt_chain_idx_m = _filter_structure(
        tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )
    qry_atom_pos_m, qry_atom_mask_m, qry_res_type_m, qry_res_idx_m, qry_chain_idx_m = _filter_structure(
        qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
    )
    tgt_q_m, _ = _filter_qscore_by_mol(
        tgt_residue_qscore, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )
    qry_q_m, _ = _filter_qscore_by_mol(
        qry_residue_qscore, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
    )

    result = _build_correspondence(
        tgt_atom_pos_m,
        tgt_atom_mask_m,
        tgt_res_type_m,
        qry_atom_pos_m,
        qry_atom_mask_m,
        qry_res_type_m,
        r=args.d,
        align=args.align,
    )

    print(f"# {mol} target_valid_residues = {len(tgt_atom_pos_m)}")
    print(f"# {mol} query_valid_residues = {len(qry_atom_pos_m)}")

    if result is None or result[0] is None:
        print(f"# {mol} no_matched_residues")
        unmatched_rows = [
            (
                "-",
                "-",
                "-",
                int(qry_chain_idx_m[i]),
                int(qry_res_idx_m[i]),
                _format_float(qry_q_m[i]),
            )
            for i in range(len(qry_atom_pos_m))
        ]
        _print_group_rows(f"{mol} unmatched", unmatched_rows)
        return

    correspondence, seq_correct_mask, matched_query_mask, _ = result
    print(f"# {mol} matched_residues = {len(correspondence)}")
    print(f"# {mol} seq_correct_residues = {int(seq_correct_mask.sum())}")

    correct_rows = []
    wrong_rows = []
    for corr_idx, (query_idx, target_idx) in enumerate(correspondence):
        row = (
            int(tgt_chain_idx_m[target_idx]),
            int(tgt_res_idx_m[target_idx]),
            _format_float(tgt_q_m[target_idx]),
            int(qry_chain_idx_m[query_idx]),
            int(qry_res_idx_m[query_idx]),
            _format_float(qry_q_m[query_idx]),
        )
        if seq_correct_mask[corr_idx]:
            correct_rows.append(row)
        else:
            wrong_rows.append(row)

    unmatched_rows = [
        (
            "-",
            "-",
            "-",
            int(qry_chain_idx_m[i]),
            int(qry_res_idx_m[i]),
            _format_float(qry_q_m[i]),
        )
        for i in range(len(qry_atom_pos_m))
        if not matched_query_mask[i]
    ]

    _print_group_rows(f"{mol} correct", correct_rows)
    _print_group_rows(f"{mol} wrong", wrong_rows)
    _print_group_rows(f"{mol} unmatched", unmatched_rows)


def main(args):
    cryo_map = load_mrc(args.map)

    tgt_polymer, tgt_atom_mask_all = _compute_residue_qscores(args.target)
    qry_polymer, qry_atom_mask_all = _compute_residue_qscores(args.query)
    tgt_residue_qscore = _calculate_residue_qscores(tgt_polymer, tgt_atom_mask_all, cryo_map)
    qry_residue_qscore = _calculate_residue_qscores(qry_polymer, qry_atom_mask_all, cryo_map)

    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx = read_pdb(
        args.target, return_bfactor=False
    )
    qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(
        args.query, return_bfactor=False
    )

    print(f"# distance_cutoff = {args.d:.4f}")
    eval_mols = (args.mol,) if args.mol is not None else ("nucleic", "protein", "complex")
    for mol in eval_mols:
        _run_one_mol(
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
            qry_residue_qscore,
            args,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
