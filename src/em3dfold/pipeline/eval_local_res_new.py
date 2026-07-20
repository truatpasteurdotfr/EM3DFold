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
from em3dfold.pipeline.eval import ALL_MOL_TYPES, PROTEIN_LABEL, _filter_structure, _select_backbone_mask
from em3dfold.utils.cryo_utils import read_map
from em3dfold.utils.geometry import kabsch

warnings.filterwarnings("ignore")

DEFAULT_LOCAL_RES_BIN_EDGES = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
STRUCT_SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")
MAP_SUFFIXES = (".mrc", ".map")


def add_args(parser):
    parser.add_argument("--list", required=True, help="Input list file, use column 1 as PDB ID")
    parser.add_argument("--target-cif-dir", required=True, help="Directory containing target PDBID.cif/.pdb files")
    parser.add_argument("--query-cif-dir", required=True, help="Directory containing query PDBID.cif/.pdb files")
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


def _compute_eval_stats(
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
    stats = _empty_stats()
    stats["n_target"] = int(len(target_atom_pos))
    stats["n_query"] = int(len(query_atom_pos))
    stats["n_cases"] = 1
    if len(target_atom_pos) == 0 or len(query_atom_pos) == 0:
        return stats

    target_anchor_pos = target_atom_pos[..., 1, :]
    query_anchor_pos = query_atom_pos[..., 1, :]
    tree = KDTree(target_anchor_pos)
    idxs = tree.query_ball_point(query_anchor_pos, r=r + 1e-3)

    sorted_idxs = []
    for q_point, target_idxs in zip(query_anchor_pos, idxs):
        target_idxs = np.asarray(target_idxs, dtype=np.int32)
        if len(target_idxs) == 0:
            sorted_idxs.append(np.zeros((0,), dtype=np.int32))
            continue
        dists = np.linalg.norm(target_anchor_pos[target_idxs] - q_point, axis=1)
        order = np.argsort(dists)
        sorted_idxs.append(target_idxs[order])

    correspondence = []
    target_idx_used = set()
    for query_idx, target_idxs in enumerate(sorted_idxs):
        chosen = None
        for target_idx in target_idxs:
            target_idx = int(target_idx)
            if target_idx in target_idx_used:
                continue
            chosen = target_idx
            break
        if chosen is not None:
            target_idx_used.add(chosen)
            correspondence.append([query_idx, chosen])

    if len(correspondence) == 0:
        return stats
    correspondence = np.asarray(correspondence, dtype=np.int32)
    stats["n_match"] = int(len(correspondence))

    query_atom_pos_eval = np.asarray(query_atom_pos, dtype=np.float32).copy()
    query_anchor_pos_eval = query_atom_pos_eval[..., 1, :]
    if align:
        R, t = kabsch(
            query_anchor_pos_eval[correspondence[:, 0]],
            target_anchor_pos[correspondence[:, 1]],
        )
        query_anchor_pos_eval = np.einsum("ij,lj->li", R, query_anchor_pos_eval) + t
        query_atom_pos_eval = np.einsum("ij,lnj->lni", R, query_atom_pos_eval) + t

    diffs = target_anchor_pos[correspondence[:, 1]] - query_anchor_pos_eval[correspondence[:, 0]]
    stats["c4_sq_sum"] = float(np.sum(np.sum(np.square(diffs), axis=-1), dtype=np.float64))

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
    if np.any(common_bb_mask):
        bb_diffs = target_bb_pos[common_bb_mask] - query_bb_pos[common_bb_mask]
        stats["bb_sq_sum"] = float(np.sum(np.sum(np.square(bb_diffs), axis=-1), dtype=np.float64))
        stats["bb_atom_count"] = int(np.sum(common_bb_mask))

    stats["n_seq_correct"] = int(
        np.sum(target_res_type[correspondence[:, 1]] == query_res_type[correspondence[:, 0]])
    )
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


def _evaluate_one_case_from_paths(mol, target_path, query_path, local_res_path, bin_specs, d, align):
    tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx = read_pdb(target_path, return_bfactor=False)
    qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx = read_pdb(query_path, return_bfactor=False)
    local_res_map, local_res_origin, local_res_voxel_size = read_map(local_res_path, ignorestart=False)

    tgt_atom_pos_m, tgt_atom_mask_m, tgt_res_type_m, _, _ = _filter_structure(
        tgt_atom_pos, tgt_atom_mask, tgt_res_type, tgt_res_idx, tgt_chain_idx, mol
    )
    qry_atom_pos_m, qry_atom_mask_m, qry_res_type_m, _, _ = _filter_structure(
        qry_atom_pos, qry_atom_mask, qry_res_type, qry_res_idx, qry_chain_idx, mol
    )

    case_stats = {
        "all": _compute_eval_stats(
            tgt_atom_pos_m,
            tgt_atom_mask_m,
            tgt_res_type_m,
            qry_atom_pos_m,
            qry_atom_mask_m,
            qry_res_type_m,
            r=d,
            mol=mol,
            align=align,
        )
    }
    for label, _, _ in bin_specs:
        case_stats[label] = _empty_stats()
        case_stats[label]["n_cases"] = 1

    if len(tgt_atom_pos_m) == 0 or len(qry_atom_pos_m) == 0:
        return case_stats

    target_anchor_pos = tgt_atom_pos_m[..., 1, :]
    target_local_res = _sample_map_values(
        target_anchor_pos,
        local_res_map,
        local_res_origin,
        local_res_voxel_size,
    ).astype(np.float32)

    for label, lower, upper in bin_specs:
        if np.isinf(upper):
            target_mask = target_local_res >= lower
        else:
            target_mask = np.logical_and(target_local_res >= lower, target_local_res < upper)
        case_stats[label] = _compute_eval_stats(
            tgt_atom_pos_m[target_mask],
            tgt_atom_mask_m[target_mask],
            tgt_res_type_m[target_mask],
            qry_atom_pos_m,
            qry_atom_mask_m,
            qry_res_type_m,
            r=d,
            mol=mol,
            align=align,
        )
    return case_stats


def _worker(task):
    pdbid, mol, target_path, query_path, local_res_path, bin_specs, d, align = task
    try:
        case_stats = _evaluate_one_case_from_paths(mol, target_path, query_path, local_res_path, bin_specs, d, align)
        return {"pdbid": pdbid, "status": "ok", "case_stats": case_stats}
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
    pdbids = _load_pdbids(args.list)
    bin_edges = _parse_local_res_bin_edges(args.local_res_bins)
    bin_specs = _build_bin_specs(bin_edges)
    eval_mols = (args.mol,) if args.mol is not None else ALL_MOL_TYPES

    print("# Using distance cutoff = {:.4f}".format(args.d))
    print("# Num cases = {}".format(len(pdbids)))
    print("# Num workers = {}".format(int(args.num_workers)))
    print("# Local resolution bins = {}".format(", ".join(label for label, _, _ in bin_specs)))

    for mol in eval_mols:
        print("#" + "-" * 72)
        print(f"# Eval {mol} (global residue-level aggregation across all cases)")
        agg = {"all": _empty_stats()}
        for label, _, _ in bin_specs:
            agg[label] = _empty_stats()

        tasks = []
        skipped = []
        for pdbid in pdbids:
            target_path = _resolve_one(args.target_cif_dir, pdbid, STRUCT_SUFFIXES)
            query_path = _resolve_one(args.query_cif_dir, pdbid, STRUCT_SUFFIXES)
            local_res_path = _resolve_one(args.local_res_map_dir, pdbid, MAP_SUFFIXES)
            missing = []
            if target_path is None:
                missing.append("target")
            if query_path is None:
                missing.append("query")
            if local_res_path is None:
                missing.append("local_res")
            if missing:
                skipped.append((pdbid, ",".join(missing)))
                continue
            tasks.append((pdbid, mol, target_path, query_path, local_res_path, bin_specs, args.d, args.align))

        if skipped:
            print(f"# skipped_missing = {len(skipped)}")
            for pdbid, reason in skipped:
                print(f"# skip {pdbid} missing={reason}")

        errored = []
        with ProcessPoolExecutor(max_workers=max(int(args.num_workers), 1)) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            ok = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"eval_local_res_new {mol}"):
                result = future.result()
                if result["status"] != "ok":
                    errored.append((result["pdbid"], result["reason"]))
                    continue
                ok += 1
                case_stats = result["case_stats"]
                _accumulate(agg["all"], case_stats["all"])
                for label, _, _ in bin_specs:
                    _accumulate(agg[label], case_stats[label])

        if errored:
            print(f"# skipped_error = {len(errored)}")
            for pdbid, reason in errored:
                print(f"# skip {pdbid} error={reason}")

        print(f"# cases_ok = {ok}")
        print(f"# cases_skipped = {len(skipped) + len(errored)}")
        _print_result(f"{mol} all", agg["all"])
        for label, _, _ in bin_specs:
            _print_result(f"{mol} bin {label}", agg[label])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
