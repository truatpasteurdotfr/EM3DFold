import argparse
import os
import warnings

import numpy as np
from scipy.spatial import KDTree

from em3dfold.io.pdbio import read_pdb

warnings.filterwarnings("ignore")


PROTEIN_LABEL = "protein"
NUCLEIC_LABEL = "nucleic"
COMPLEX_LABEL = "complex"
ALL_MOL_TYPES = (PROTEIN_LABEL, NUCLEIC_LABEL, COMPLEX_LABEL)


def kabsch(P: np.ndarray, Q: np.ndarray):
    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have the same shape, got {P.shape} and {Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P and Q must have shape (N, 3), got {P.shape}")
    if P.shape[0] < 1:
        raise ValueError("Need at least one point")

    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_Q - centroid_P @ R.T
    P_aligned = P @ R.T + t
    return R, t, P_aligned


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
    return parser


def _normalize_na_restypes(res_type):
    res_type = np.asarray(res_type, dtype=np.int32).copy()
    res_type[res_type == 20] = 24
    res_type[res_type == 21] = 25
    res_type[res_type == 22] = 26
    res_type[res_type == 23] = 27
    return res_type


def _residue_mask(res_type, atom_mask, mol):
    has_backbone_anchor = atom_mask[..., 1].astype(bool)
    if mol == PROTEIN_LABEL:
        return np.logical_and(has_backbone_anchor, res_type < 20)
    if mol == NUCLEIC_LABEL:
        return np.logical_and(has_backbone_anchor, res_type >= 20)
    return has_backbone_anchor


def _filter_structure(atom_pos, atom_mask, res_type, res_idx, chain_idx, mol):
    keep_mask = _residue_mask(res_type, atom_mask, mol)
    return (
        atom_pos[keep_mask],
        atom_mask[keep_mask],
        _normalize_na_restypes(res_type[keep_mask]),
        res_idx[keep_mask],
        chain_idx[keep_mask],
    )


def _format_metrics(metrics):
    if metrics is None:
        return "CX_RMSD = - BB_RMSD = - Cov = 0.0 Seq_Match = - Seq_Recall = -"
    return (
        "CX_RMSD = {:.4f} BB_RMSD = {:.4f} Cov = {:.4f} Seq_Match = {:.4f} Seq_Recall = {:.4f}".format(
            metrics["c4_rmsd"],
            metrics["bb_rmsd"],
            metrics["cov"],
            metrics["seq_match"],
            metrics["seq_recall"],
        )
    )


def _select_backbone_mask(atom_mask, res_type, mol):
    atom_mask = np.asarray(atom_mask).copy()
    if mol == PROTEIN_LABEL:
        atom_mask[..., 4:] = False
        return atom_mask
    if mol == NUCLEIC_LABEL:
        atom_mask[..., 11:] = False
        return atom_mask

    protein_rows = res_type < 20
    nucleic_rows = np.logical_not(protein_rows)
    if np.any(protein_rows):
        atom_mask[protein_rows, 4:] = False
    if np.any(nucleic_rows):
        atom_mask[nucleic_rows, 11:] = False
    return atom_mask


def eval_local(
    target_atom_pos,
    target_atom_mask,
    target_res_type,
    query_atom_pos,
    query_atom_mask,
    query_res_type,
    r=3.0,
    mol=PROTEIN_LABEL,
    align=False,
):
    if len(target_atom_pos) == 0 or len(query_atom_pos) == 0:
        return None

    target_anchor_pos = target_atom_pos[..., 1, :]
    query_anchor_pos = query_atom_pos[..., 1, :]
    tree = KDTree(target_anchor_pos)

    idxs = tree.query_ball_point(query_anchor_pos, r=r + 1e-3)

    sorted_idxs = []
    sorted_distances = []
    for q_point, target_idxs in zip(query_anchor_pos, idxs):
        target_idxs = np.asarray(target_idxs, dtype=np.int32)

        if len(target_idxs) == 0:
            sorted_idxs.append([])
            sorted_distances.append([])
            continue

        dists = np.linalg.norm(target_anchor_pos[target_idxs] - q_point, axis=1)
        order = np.argsort(dists)
        sorted_idxs.append(np.asarray(target_idxs)[order])
        sorted_distances.append(dists[order])

    correspondence = []
    target_idx_used = set()
    for query_idx, (target_idxs, _) in enumerate(zip(sorted_idxs, sorted_distances)):
        unused_idx = None
        for target_idx in target_idxs:
            if target_idx in target_idx_used:
                continue
            unused_idx = target_idx
            break
        if unused_idx is not None:
            target_idx_used.add(unused_idx)
            correspondence.append([query_idx, unused_idx])

    correspondence = np.asarray(correspondence, dtype=np.int32)
    if len(correspondence) == 0:
        return None

    query_atom_pos_eval = np.asarray(query_atom_pos, dtype=np.float32).copy()
    query_anchor_pos_eval = query_atom_pos_eval[..., 1, :]

    if align:
        print("# Align mode")
        R, t, _ = kabsch(
            query_anchor_pos_eval[correspondence[:, 0]],
            target_anchor_pos[correspondence[:, 1]],
        )
        query_anchor_pos_eval = np.einsum("ij,lj->li", R, query_anchor_pos_eval) + t
        query_atom_pos_eval = np.einsum("ij,lnj->lni", R, query_atom_pos_eval) + t

    c4_rmsd = np.sqrt(
        np.mean(
            np.sum(
                np.power(
                    target_anchor_pos[correspondence[:, 1]] - query_anchor_pos_eval[correspondence[:, 0]],
                    2,
                ),
                axis=-1,
            ),
        ),
    )

    target_bb_pos = target_atom_pos[correspondence[:, 1]].copy()
    target_bb_mask = _select_backbone_mask(
        target_atom_mask[correspondence[:, 1]],
        target_res_type[correspondence[:, 1]],
        mol,
    )

    query_bb_pos = query_atom_pos_eval[correspondence[:, 0]].copy()
    query_bb_mask = _select_backbone_mask(
        query_atom_mask[correspondence[:, 0]],
        query_res_type[correspondence[:, 0]],
        mol,
    )

    common_bb_mask = np.logical_and(target_bb_mask, query_bb_mask)
    if not np.any(common_bb_mask):
        bb_rmsd = float("nan")
    else:
        bb_rmsd = np.sqrt(
            np.mean(
                np.sum(
                    np.power(target_bb_pos[common_bb_mask] - query_bb_pos[common_bb_mask], 2),
                    axis=-1,
                )
            ),
        )

    cov = len(correspondence[:, 0]) / max(len(target_anchor_pos), 1)
    seq_match = np.sum(
        target_res_type[correspondence[:, 1]] == query_res_type[correspondence[:, 0]]
    ) / len(correspondence[:, 0])
    seq_recall = cov * seq_match

    return {
        "n_query": int(len(query_atom_pos)),
        "n_target": int(len(target_atom_pos)),
        "n_match": int(len(correspondence[:, 0])),
        "c4_rmsd": float(c4_rmsd),
        "bb_rmsd": float(bb_rmsd),
        "cov": float(cov),
        "seq_match": float(seq_match),
        "seq_recall": float(seq_recall),
    }


def _print_eval_result(scope_name, metrics, query_len):
    if metrics is None:
        print(f"# {scope_name} len = {query_len} CX_RMSD = - BB_RMSD = - Cov = 0.0 Seq_Match = - Seq_Recall = -")
        return
    print(
        "# {} len = {} CX_RMSD = {:.4f} BB_RMSD = {:.4f} Cov = {:.4f} Seq_Match = {:.4f} Seq_Recall = {:.4f}".format(
            scope_name,
            query_len,
            metrics["c4_rmsd"],
            metrics["bb_rmsd"],
            metrics["cov"],
            metrics["seq_match"],
            metrics["seq_recall"],
        )
    )


def _evaluate_one_mol(
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

    print("#" + "-" * 72)
    print(f"# Eval {mol}")
    print(f"# Read {len(tgt_atom_pos_m)} valid residues from {target_path}")
    print(f"# Read {len(qry_atom_pos_m)} valid residues from {query_path}")

    if len(tgt_atom_pos_m) == 0 or len(qry_atom_pos_m) == 0:
        print(f"# Skip {mol} evaluation because target or query has no valid residues")
        return

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


def main(args):
    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx = read_pdb(
        args.target, return_bfactor=False
    )
    qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(
        args.query, return_bfactor=False
    )

    print("# Using distance cutoff = {:.4f}".format(args.d))
    eval_mols = (args.mol,) if args.mol is not None else ALL_MOL_TYPES

    for mol in eval_mols:
        _evaluate_one_mol(
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
            args,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
