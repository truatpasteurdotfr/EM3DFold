import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree
from tqdm import tqdm

from em3dfold.io.pdbio import read_pdb
from em3dfold.pipeline.eval import ALL_MOL_TYPES, NUCLEIC_LABEL, PROTEIN_LABEL, _filter_structure, _select_backbone_mask
from em3dfold.utils.cryo_utils import read_map
from em3dfold.utils.geometry import kabsch

warnings.filterwarnings("ignore")

DEFAULT_LOCAL_RES_BIN_EDGES = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
STRUCT_SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")
MAP_SUFFIXES = (".mrc", ".map")


def add_args(parser):
    parser.add_argument("--list", required=True, help="Input list file, use column 1 as PDB ID")
    parser.add_argument("--target-cif-dir", required=True, help="Directory containing target PDBID.cif/.pdb files")
    parser.add_argument(
        "--query-cif-dir",
        nargs="+",
        required=True,
        help="One or more directories containing query PDBID.cif/.pdb files",
    )
    parser.add_argument(
        "--query-label",
        nargs="+",
        default=None,
        help="Optional labels for query dirs, same length as --query-cif-dir",
    )
    parser.add_argument(
        "--local-res-map-dir",
        required=True,
        help="Directory containing local resolution PDBID.mrc/.map files",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
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
    lower = max(0.0, float(edges[0]) - 0.5)
    for upper in edges:
        label = f"{_format_bin_value(lower)}-{_format_bin_value(upper)}"
        specs.append((label, lower, float(upper)))
        lower = float(upper)
    upper = float(edges[-1]) + 0.5
    specs.append((f"{_format_bin_value(edges[-1])}-{_format_bin_value(upper)}", float(edges[-1]), upper))
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


def _empty_stats():
    return {
        "n_target": 0,
        "n_query": 0,
        "n_match": 0,
        "n_seq_correct": 0,
        "c4_sq_sum": 0.0,
        "bb_sq_sum": 0.0,
        "bb_atom_count": 0,
        "n_cases": 0,
    }


def _accumulate(dst, src):
    for key in dst:
        dst[key] += src[key]


def _stats_to_metrics(stats):
    if stats["n_target"] == 0 or stats["n_match"] == 0:
        return None
    c4_rmsd = float(np.sqrt(stats["c4_sq_sum"] / stats["n_match"]))
    if stats["bb_atom_count"] > 0:
        bb_rmsd = float(np.sqrt(stats["bb_sq_sum"] / stats["bb_atom_count"]))
    else:
        bb_rmsd = float("nan")
    cov = float(stats["n_match"] / max(stats["n_target"], 1))
    seq_match = float(stats["n_seq_correct"] / max(stats["n_match"], 1))
    seq_recall = float(stats["n_seq_correct"] / max(stats["n_target"], 1))
    return {
        "c4_rmsd": c4_rmsd,
        "bb_rmsd": bb_rmsd,
        "cov": cov,
        "seq_match": seq_match,
        "seq_recall": seq_recall,
    }


def _default_query_labels(query_dirs):
    return [str(i) for i in range(len(query_dirs))]


def _compute_correspondence(target_anchor_pos, query_anchor_pos, cutoff):
    tree = KDTree(target_anchor_pos)
    idxs = tree.query_ball_point(query_anchor_pos, r=cutoff + 1e-3)

    sorted_idxs = []
    for q_point, target_idxs in zip(query_anchor_pos, idxs):
        target_idxs = np.asarray(target_idxs, dtype=np.int32)
        if len(target_idxs) == 0:
            sorted_idxs.append(np.zeros((0,), dtype=np.int32))
            continue
        dists = np.linalg.norm(target_anchor_pos[target_idxs] - q_point, axis=1)
        order = np.argsort(dists)
        sorted_idxs.append(target_idxs[order])

    query_to_target = np.full((len(query_anchor_pos),), -1, dtype=np.int32)
    used = set()
    for query_idx, target_idxs in enumerate(sorted_idxs):
        for target_idx in target_idxs:
            target_idx = int(target_idx)
            if target_idx in used:
                continue
            used.add(target_idx)
            query_to_target[query_idx] = target_idx
            break
    return query_to_target


def _build_target_to_query(query_to_target, n_target):
    target_to_query = np.full((n_target,), -1, dtype=np.int32)
    matched_query_rows = np.nonzero(query_to_target >= 0)[0].astype(np.int32)
    target_to_query[query_to_target[matched_query_rows]] = matched_query_rows
    return target_to_query


def _compute_subset_stats(
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
    stats = _empty_stats()
    stats["n_target"] = int(np.sum(subset_target_mask))
    stats["n_query"] = int(len(query_atom_pos))
    stats["n_cases"] = 1

    subset_target_indices = np.nonzero(subset_target_mask)[0].astype(np.int32)
    if len(subset_target_indices) == 0:
        return stats

    subset_query_indices = target_to_query[subset_target_indices]
    keep = subset_query_indices >= 0
    subset_target_indices = subset_target_indices[keep]
    subset_query_indices = subset_query_indices[keep]
    stats["n_match"] = int(len(subset_target_indices))
    if len(subset_target_indices) == 0:
        return stats

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

    diffs = target_anchor_pos[subset_target_indices] - query_anchor_pos[subset_query_indices]
    stats["c4_sq_sum"] = float(np.sum(np.sum(np.square(diffs), axis=-1), dtype=np.float64))

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
    if np.any(common_bb_mask):
        bb_diffs = target_bb_pos[common_bb_mask] - query_bb_pos[common_bb_mask]
        stats["bb_sq_sum"] = float(np.sum(np.sum(np.square(bb_diffs), axis=-1), dtype=np.float64))
        stats["bb_atom_count"] = int(np.sum(common_bb_mask))

    stats["n_seq_correct"] = int(np.sum(target_res_type[subset_target_indices] == query_res_type[subset_query_indices]))
    return stats


def _resolve_one(base_dir, pdbid, suffixes):
    base = Path(base_dir)
    for suffix in suffixes:
        path = base / f"{pdbid}{suffix}"
        if path.is_file():
            return str(path.resolve())
    return None


def _load_pdbids(list_path):
    pdbids = []
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pdbids.append(line.split()[0])
    return pdbids


def _evaluate_one_case_common(mol, target_path, query_paths, local_res_path, bin_specs, d, align, query_labels):
    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx = read_pdb(target_path, return_bfactor=False)
    tgt_atom_pos_m, tgt_atom_mask_m, tgt_res_type_m, _, _ = _filter_structure(
        tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )
    local_res_map, local_res_origin, local_res_voxel_size = read_map(local_res_path, ignorestart=False)

    query_infos = []
    target_anchor_pos = tgt_atom_pos_m[..., 1, :] if len(tgt_atom_pos_m) > 0 else np.zeros((0, 3), dtype=np.float32)
    for label, query_path in zip(query_labels, query_paths):
        qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(query_path, return_bfactor=False)
        qry_atom_pos_m, qry_atom_mask_m, qry_res_type_m, _, _ = _filter_structure(
            qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
        )
        if len(tgt_atom_pos_m) == 0 or len(qry_atom_pos_m) == 0:
            target_to_query = np.full((len(tgt_atom_pos_m),), -1, dtype=np.int32)
            matched_target_mask = np.zeros((len(tgt_atom_pos_m),), dtype=bool)
        else:
            query_to_target = _compute_correspondence(target_anchor_pos, qry_atom_pos_m[..., 1, :], cutoff=d)
            target_to_query = _build_target_to_query(query_to_target, len(tgt_atom_pos_m))
            matched_target_mask = target_to_query >= 0
        query_infos.append(
            {
                "label": label,
                "atom_pos": qry_atom_pos_m,
                "atom_mask": qry_atom_mask_m,
                "res_type": qry_res_type_m,
                "target_to_query": target_to_query,
                "matched_target_mask": matched_target_mask,
            }
        )

    shared_mask = np.zeros((len(tgt_atom_pos_m),), dtype=bool)
    if query_infos:
        matched_matrix = np.stack([info["matched_target_mask"] for info in query_infos], axis=0)
        shared_mask = np.sum(matched_matrix, axis=0) == len(query_infos)

    target_local_res = np.zeros((len(tgt_atom_pos_m),), dtype=np.float32)
    if len(tgt_atom_pos_m) > 0:
        target_local_res = _sample_map_values(
            target_anchor_pos,
            local_res_map,
            local_res_origin,
            local_res_voxel_size,
        ).astype(np.float32)

    results = {}
    for info in query_infos:
        per_query = {"all": _compute_subset_stats(
            tgt_atom_pos_m,
            tgt_atom_mask_m,
            tgt_res_type_m,
            info["atom_pos"],
            info["atom_mask"],
            info["res_type"],
            info["target_to_query"],
            shared_mask,
            mol,
            align=align,
        )}
        for bin_label, lower, upper in bin_specs:
            bin_mask = np.logical_and(target_local_res >= lower, target_local_res < upper)
            subset_mask = np.logical_and(shared_mask, bin_mask)
            per_query[bin_label] = _compute_subset_stats(
                tgt_atom_pos_m,
                tgt_atom_mask_m,
                tgt_res_type_m,
                info["atom_pos"],
                info["atom_mask"],
                info["res_type"],
                info["target_to_query"],
                subset_mask,
                mol,
                align=align,
            )
        results[info["label"]] = per_query
    return results


def _worker(task):
    pdbid, mol, target_path, query_paths, local_res_path, bin_specs, d, align, query_labels = task
    try:
        results = _evaluate_one_case_common(mol, target_path, query_paths, local_res_path, bin_specs, d, align, query_labels)
        return {"pdbid": pdbid, "status": "ok", "results": results}
    except Exception as exc:
        return {"pdbid": pdbid, "status": "error", "reason": str(exc)}


def _print_result(scope_name, stats):
    metrics = _stats_to_metrics(stats)
    if metrics is None:
        print(
            f"# {scope_name} cases = {stats['n_cases']} target_len = {stats['n_target']} match_len = {stats['n_match']} "
            f"CX_RMSD = - BB_RMSD = - Cov = 0.0 Seq_Match = - Seq_Recall = -"
        )
        return
    print(
        "# {} cases = {} target_len = {} match_len = {} CX_RMSD = {:.4f} BB_RMSD = {:.4f} Cov = {:.4f} "
        "Seq_Match = {:.4f} Seq_Recall = {:.4f}".format(
            scope_name,
            stats["n_cases"],
            stats["n_target"],
            stats["n_match"],
            metrics["c4_rmsd"],
            metrics["bb_rmsd"],
            metrics["cov"],
            metrics["seq_match"],
            metrics["seq_recall"],
        )
    )


def main(args):
    if args.query_label is not None and len(args.query_label) != len(args.query_cif_dir):
        raise ValueError("--query-label must have the same length as --query-cif-dir")
    query_labels = args.query_label if args.query_label is not None else _default_query_labels(args.query_cif_dir)
    pdbids = _load_pdbids(args.list)
    bin_edges = _parse_local_res_bin_edges(args.local_res_bins)
    bin_specs = _build_bin_specs(bin_edges)
    eval_mols = (args.mol,) if args.mol is not None else ALL_MOL_TYPES

    print("# Using distance cutoff = {:.4f}".format(args.d))
    print("# Num cases = {}".format(len(pdbids)))
    print("# Num query dirs = {}".format(len(args.query_cif_dir)))
    print("# Query labels = {}".format(", ".join(query_labels)))
    print("# Num workers = {}".format(int(args.num_workers)))
    print("# Local resolution bins = {}".format(", ".join(label for label, _, _ in bin_specs)))

    for mol in eval_mols:
        print("#" + "-" * 72)
        print(f"# Eval {mol} common residues (global aggregation across all cases)")
        agg = {
            label: {"all": _empty_stats(), **{bin_label: _empty_stats() for bin_label, _, _ in bin_specs}}
            for label in query_labels
        }

        tasks = []
        skipped = []
        for pdbid in pdbids:
            target_path = _resolve_one(args.target_cif_dir, pdbid, STRUCT_SUFFIXES)
            local_res_path = _resolve_one(args.local_res_map_dir, pdbid, MAP_SUFFIXES)
            query_paths = []
            missing = []
            if target_path is None:
                missing.append("target")
            if local_res_path is None:
                missing.append("local_res")
            for label, query_dir in zip(query_labels, args.query_cif_dir):
                query_path = _resolve_one(query_dir, pdbid, STRUCT_SUFFIXES)
                if query_path is None:
                    missing.append(f"query:{label}")
                query_paths.append(query_path)
            if missing:
                skipped.append((pdbid, ",".join(missing)))
                continue
            tasks.append((pdbid, mol, target_path, query_paths, local_res_path, bin_specs, args.d, args.align, query_labels))

        if skipped:
            print(f"# skipped_missing = {len(skipped)}")
            for pdbid, reason in skipped:
                print(f"# skip {pdbid} missing={reason}")

        errored = []
        ok = 0
        with ProcessPoolExecutor(max_workers=max(int(args.num_workers), 1)) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"eval_local_res_new_common {mol}"):
                result = future.result()
                if result["status"] != "ok":
                    errored.append((result["pdbid"], result["reason"]))
                    continue
                ok += 1
                for label in query_labels:
                    _accumulate(agg[label]["all"], result["results"][label]["all"])
                    for bin_label, _, _ in bin_specs:
                        _accumulate(agg[label][bin_label], result["results"][label][bin_label])

        if errored:
            print(f"# skipped_error = {len(errored)}")
            for pdbid, reason in errored:
                print(f"# skip {pdbid} error={reason}")

        print(f"# cases_ok = {ok}")
        print(f"# cases_skipped = {len(skipped) + len(errored)}")
        for label in query_labels:
            _print_result(f"{mol} shared {label} all", agg[label]["all"])
            for bin_label, _, _ in bin_specs:
                _print_result(f"{mol} shared {label} bin {bin_label}", agg[label][bin_label])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
