import argparse
import os
import warnings

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree

from em3dfold.io.pdbio import read_pdb
from em3dfold.pipeline.eval import ALL_MOL_TYPES, NUCLEIC_LABEL, PROTEIN_LABEL, _filter_structure
from em3dfold.utils.cryo_utils import read_map
from em3dfold.utils.geometry import kabsch

warnings.filterwarnings("ignore")

DEFAULT_LOCAL_RES_BIN_EDGES = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def add_args(parser):
    parser.add_argument("--target", "-t", required=True, help="Target structure")
    parser.add_argument("--query", "-q", nargs="+", required=True, help="Two or more query structures")
    parser.add_argument(
        "--query-label",
        nargs="+",
        default=None,
        help="Optional labels for queries, same length as --query",
    )
    parser.add_argument("--local-res-map", required=True, help="Local resolution map in MRC/MAP format")
    parser.add_argument("-d", type=float, default=3.0, help="Distance cutoff")
    parser.add_argument("--align", action="store_true", default=False)
    parser.add_argument(
        "--mol",
        choices=ALL_MOL_TYPES,
        default=None,
        help="If omitted, evaluate protein, nucleic and complex.",
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


def _default_query_labels(query_paths):
    return [str(idx) for idx, _path in enumerate(query_paths)]


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


def _compute_correspondence(target_anchor_pos, query_anchor_pos, cutoff):
    tree = KDTree(target_anchor_pos)
    idxs = tree.query_ball_point(query_anchor_pos, r=cutoff + 1e-3)

    sorted_idxs = []
    sorted_distances = []
    for q_point, target_idxs in zip(query_anchor_pos, idxs):
        target_idxs = np.asarray(target_idxs, dtype=np.int32)
        if len(target_idxs) == 0:
            sorted_idxs.append(np.zeros((0,), dtype=np.int32))
            sorted_distances.append(np.zeros((0,), dtype=np.float32))
            continue
        dists = np.linalg.norm(target_anchor_pos[target_idxs] - q_point, axis=1)
        order = np.argsort(dists)
        sorted_idxs.append(target_idxs[order])
        sorted_distances.append(dists[order].astype(np.float32))

    query_to_target = np.full((len(query_anchor_pos),), -1, dtype=np.int32)
    query_to_distance = np.full((len(query_anchor_pos),), np.nan, dtype=np.float32)
    target_idx_used = set()
    for query_idx, (target_idxs, dists) in enumerate(zip(sorted_idxs, sorted_distances)):
        for target_idx, dist in zip(target_idxs, dists):
            target_idx = int(target_idx)
            if target_idx in target_idx_used:
                continue
            target_idx_used.add(target_idx)
            query_to_target[query_idx] = target_idx
            query_to_distance[query_idx] = float(dist)
            break
    return query_to_target, query_to_distance


def _build_target_to_query(query_to_target, n_target):
    target_to_query = np.full((n_target,), -1, dtype=np.int32)
    matched_query_rows = np.nonzero(query_to_target >= 0)[0].astype(np.int32)
    target_to_query[query_to_target[matched_query_rows]] = matched_query_rows
    return target_to_query


def _compute_subset_metrics(
    target_atom_pos,
    target_atom_mask,
    target_res_type,
    query_atom_pos,
    query_atom_mask,
    query_res_type,
    target_to_query,
    subset_target_mask,
    mol,
    align=False,
):
    subset_target_indices = np.nonzero(subset_target_mask)[0].astype(np.int32)
    if len(subset_target_indices) == 0:
        return None

    subset_query_indices = target_to_query[subset_target_indices]
    if np.any(subset_query_indices < 0):
        raise RuntimeError("subset contains residues missing in query")

    query_atom_pos_eval = np.asarray(query_atom_pos, dtype=np.float32).copy()
    target_anchor_pos = target_atom_pos[..., 1, :]
    query_anchor_pos = query_atom_pos_eval[..., 1, :]

    if align:
        R, t = kabsch(
            query_anchor_pos[subset_query_indices],
            target_anchor_pos[subset_target_indices],
        )
        query_atom_pos_eval = np.einsum("ij,lnj->lni", R, query_atom_pos_eval) + t
        query_anchor_pos = query_atom_pos_eval[..., 1, :]

    c4_rmsd = np.sqrt(
        np.mean(
            np.sum(
                np.power(
                    target_anchor_pos[subset_target_indices] - query_anchor_pos[subset_query_indices],
                    2,
                ),
                axis=-1,
            ),
        ),
    )

    target_bb_pos = target_atom_pos[subset_target_indices].copy()
    target_bb_mask = _select_backbone_mask(
        target_atom_mask[subset_target_indices],
        target_res_type[subset_target_indices],
        mol,
    )
    query_bb_pos = query_atom_pos_eval[subset_query_indices].copy()
    query_bb_mask = _select_backbone_mask(
        query_atom_mask[subset_query_indices],
        query_res_type[subset_query_indices],
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
                ),
            ),
        )

    cov = len(subset_target_indices) / max(len(target_atom_pos), 1)
    seq_match = np.sum(
        target_res_type[subset_target_indices] == query_res_type[subset_query_indices]
    ) / len(subset_target_indices)
    seq_recall = cov * seq_match

    return {
        "n_query": int(len(query_atom_pos)),
        "n_target": int(len(target_atom_pos)),
        "n_subset": int(len(subset_target_indices)),
        "c4_rmsd": float(c4_rmsd),
        "bb_rmsd": float(bb_rmsd),
        "cov": float(cov),
        "seq_match": float(seq_match),
        "seq_recall": float(seq_recall),
    }


def _format_metrics(metrics):
    if metrics is None:
        return "n = 0 CX_RMSD = - BB_RMSD = - Cov = 0.0 Seq_Match = - Seq_Recall = -"
    return (
        "n = {n_subset} CX_RMSD = {c4_rmsd:.4f} BB_RMSD = {bb_rmsd:.4f} "
        "Cov = {cov:.4f} Seq_Match = {seq_match:.4f} Seq_Recall = {seq_recall:.4f}"
    ).format(**metrics)


def _evaluate_one_mol_local_res_common(
    mol,
    tgt_atom_pos,
    tgt_atom_mask,
    tgt_res_type,
    tgt_res_idx,
    tgt_chain_idx,
    query_paths,
    query_labels,
    local_res_map,
    local_res_origin,
    local_res_voxel_size,
    bin_specs,
    args,
):
    target_path = os.path.abspath(args.target)
    local_res_path = os.path.abspath(args.local_res_map)
    tgt_atom_pos_m, tgt_atom_mask_m, tgt_res_type_m, tgt_res_idx_m, tgt_chain_idx_m = _filter_structure(
        tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )

    print("#" + "-" * 72)
    print(f"# Eval shared local-res {mol}")
    print(f"# Read {len(tgt_atom_pos_m)} valid residues from {target_path}")
    print(f"# Read local resolution map from {local_res_path}")

    if len(tgt_atom_pos_m) == 0:
        print(f"# Skip {mol} evaluation because target has no valid residues")
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

    query_infos = []
    for query_idx, query_path in enumerate(query_paths):
        qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(
            query_path,
            return_bfactor=False,
        )
        qry_atom_pos_m, qry_atom_mask_m, qry_res_type_m, qry_res_idx_m, qry_chain_idx_m = _filter_structure(
            qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
        )
        print(
            f"# Read {len(qry_atom_pos_m)} valid residues from {os.path.abspath(query_path)} "
            f"[{query_labels[query_idx]}]"
        )

        if len(qry_atom_pos_m) == 0:
            target_to_query = np.full((len(tgt_atom_pos_m),), -1, dtype=np.int32)
            matched_target_mask = np.zeros((len(tgt_atom_pos_m),), dtype=bool)
        else:
            query_to_target, _query_to_distance = _compute_correspondence(
                target_anchor_pos,
                qry_atom_pos_m[..., 1, :],
                cutoff=args.d,
            )
            target_to_query = _build_target_to_query(query_to_target, len(tgt_atom_pos_m))
            matched_target_mask = target_to_query >= 0

        query_infos.append(
            {
                "query_index": query_idx,
                "query_label": query_labels[query_idx],
                "query_path": query_path,
                "atom_pos": qry_atom_pos_m,
                "atom_mask": qry_atom_mask_m,
                "res_type": qry_res_type_m,
                "target_to_query": target_to_query,
                "matched_target_mask": matched_target_mask,
            }
        )

    matched_matrix = np.stack([info["matched_target_mask"] for info in query_infos], axis=0)
    shared_mask = np.sum(matched_matrix, axis=0) == len(query_infos)
    print(f"# shared target residues = {int(np.sum(shared_mask))}")

    for info in query_infos:
        shared_metrics = _compute_subset_metrics(
            tgt_atom_pos_m,
            tgt_atom_mask_m,
            tgt_res_type_m,
            info["atom_pos"],
            info["atom_mask"],
            info["res_type"],
            info["target_to_query"],
            shared_mask,
            mol,
            align=args.align,
        )
        print(f"# shared {info['query_label']} {_format_metrics(shared_metrics)}")

    for label, lower, upper in bin_specs:
        if np.isinf(upper):
            bin_mask = target_local_res >= lower
        else:
            bin_mask = np.logical_and(target_local_res >= lower, target_local_res < upper)
        subset_mask = np.logical_and(shared_mask, bin_mask)
        print(f"# shared local-res bin {label} target_len = {int(np.sum(subset_mask))}")
        for info in query_infos:
            bin_metrics = _compute_subset_metrics(
                tgt_atom_pos_m,
                tgt_atom_mask_m,
                tgt_res_type_m,
                info["atom_pos"],
                info["atom_mask"],
                info["res_type"],
                info["target_to_query"],
                subset_mask,
                mol,
                align=args.align,
            )
            print(f"# shared {info['query_label']} bin {label} {_format_metrics(bin_metrics)}")


def main(args):
    if len(args.query) < 2:
        raise ValueError("Need at least two query structures")
    if args.query_label is not None and len(args.query_label) != len(args.query):
        raise ValueError("--query-label must have the same length as --query")

    query_labels = args.query_label if args.query_label is not None else _default_query_labels(args.query)
    bin_edges = _parse_local_res_bin_edges(args.local_res_bins)
    bin_specs = _build_bin_specs(bin_edges)

    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx = read_pdb(
        args.target,
        return_bfactor=False,
    )
    local_res_map, local_res_origin, local_res_voxel_size = read_map(args.local_res_map, ignorestart=False)

    print("# Using distance cutoff = {:.4f}".format(args.d))
    print("# Num queries = {}".format(len(args.query)))
    print("# Query labels = {}".format(", ".join(query_labels)))
    print("# Local resolution bins = {}".format(", ".join(label for label, _, _ in bin_specs)))
    eval_mols = (args.mol,) if args.mol is not None else ALL_MOL_TYPES

    for mol in eval_mols:
        _evaluate_one_mol_local_res_common(
            mol,
            tgt_atom_pos,
            tgt_atom_mask,
            tgt_res_type,
            tgt_res_idx,
            tgt_chain_idx,
            args.query,
            query_labels,
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
