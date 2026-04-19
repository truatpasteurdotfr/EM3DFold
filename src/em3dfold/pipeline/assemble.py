"""Assemble chains by selecting a non-clashing high-scoring protein subset.

This entrypoint scores every protein chain against an input ``ca.mrc`` map,
computes pairwise protein-protein clashes, and then uses OR-Tools CP-SAT to
select the maximum-score compatible subset. Nucleic-acid chains are kept by
default and do not participate in scoring or clash detection.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

from em3dfold.io.pdbio import chains_atom_pos_to_pdb, convert_to_chains, read_pdb
from em3dfold.utils.clash_utils import get_clash
from em3dfold.utils.cryo_utils import read_map
from em3dfold.utils.misc_utils import abspath, pjoin


SCORE_SCALE = 1000


def print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    message = sep.join(str(arg) for arg in args)
    if not message.startswith("# "):
        message = f"# {message}"
    builtins.print(message, **kwargs)


@dataclass(slots=True)
class ChainRecord:
    global_index: int
    input_path: str
    input_file_index: int
    source_chain_index: int
    chain_type: str
    atom_pos: np.ndarray
    atom_mask: np.ndarray
    res_type: np.ndarray
    res_idx: np.ndarray
    bfactor: np.ndarray
    residue_count: int
    ca_count: int
    score: float | None = None
    keep: bool = False
    keep_reason: str | None = None
    clash_partners: list[int] = field(default_factory=list)


def _resolve_structure_paths(input_paths):
    resolved_paths = []
    for input_path in input_paths:
        input_path = abspath(input_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input structure path not found: {input_path}")

        if os.path.isdir(input_path):
            local_paths = []
            for suffix in ("*.pdb", "*.cif", "*.ent", "*.mmcif"):
                local_paths.extend(sorted(str(path) for path in Path(input_path).glob(suffix)))
            if not local_paths:
                raise FileNotFoundError(f"No supported structure files found under {input_path}")
            resolved_paths.extend(local_paths)
            continue

        resolved_paths.append(input_path)

    if not resolved_paths:
        raise ValueError("No input structures were resolved.")
    return resolved_paths


def _classify_chain(chain_res_type):
    if len(chain_res_type) == 0:
        return "other"

    chain_res_type = np.asarray(chain_res_type, dtype=np.int32)
    is_protein = np.all(chain_res_type < 20)
    is_nucleic = np.all(chain_res_type >= 20)
    if is_protein:
        return "protein"
    if is_nucleic:
        return "nucleic"
    return "mixed"


def _ca_positions(chain_record):
    ca_mask = np.asarray(chain_record.atom_mask[:, 1] > 0, dtype=bool)
    if not np.any(ca_mask):
        return np.zeros((0, 3), dtype=np.float32)
    coords = np.asarray(chain_record.atom_pos[ca_mask, 1, :], dtype=np.float32)
    finite_mask = np.all(np.isfinite(coords), axis=1)
    return coords[finite_mask]


def _load_chain_records(structure_paths):
    chain_records = []
    for input_file_index, structure_path in enumerate(structure_paths):
        print(f"Read structure {structure_path}")
        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
            structure_path,
            keep_valid=False,
            return_bfactor=True,
        )
        (
            chains_atom_pos,
            chains_atom_mask,
            chains_res_type,
            chains_res_idx,
            chains_bfactor,
        ) = convert_to_chains(
            chain_idx,
            atom_pos,
            atom_mask,
            res_type,
            res_idx,
            bfactor,
        )

        for source_chain_index in range(len(chains_atom_pos)):
            chain_res_type = np.asarray(chains_res_type[source_chain_index], dtype=np.int32)
            chain_record = ChainRecord(
                global_index=len(chain_records),
                input_path=structure_path,
                input_file_index=input_file_index,
                source_chain_index=source_chain_index,
                chain_type=_classify_chain(chain_res_type),
                atom_pos=np.asarray(chains_atom_pos[source_chain_index], dtype=np.float32),
                atom_mask=np.asarray(chains_atom_mask[source_chain_index], dtype=np.int32),
                res_type=chain_res_type,
                res_idx=np.asarray(chains_res_idx[source_chain_index], dtype=np.int32),
                bfactor=np.asarray(chains_bfactor[source_chain_index], dtype=np.float32),
                residue_count=len(chain_res_type),
                ca_count=0,
            )
            chain_record.ca_count = len(_ca_positions(chain_record))
            chain_records.append(chain_record)

            print(
                "  chain {} -> type={} residues={} CA={}".format(
                    source_chain_index,
                    chain_record.chain_type,
                    chain_record.residue_count,
                    chain_record.ca_count,
                )
            )

    if not chain_records:
        raise ValueError("No chains were parsed from the input structures.")
    return chain_records


def _read_ca_map(map_path):
    data, origin, voxel_size = read_map(map_path, ignorestart=False)
    data = np.asarray(data, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    if voxel_size.shape != (3,) or np.any(~np.isfinite(voxel_size)) or np.any(voxel_size <= 0):
        raise ValueError(f"Invalid voxel size in map header: {voxel_size}")
    return data, origin, voxel_size


def _normalize_density_map(data, percentile):
    data = np.asarray(data, dtype=np.float32)
    density_cap = float(np.percentile(data, q=percentile))
    if density_cap <= 0:
        density_cap = float(np.max(data))
    if density_cap <= 0:
        return np.zeros_like(data, dtype=np.float32)
    normalized = np.clip(data, 0.0, density_cap) / (density_cap + 1e-6)
    return (normalized * 10.0).astype(np.float32, copy=False)


def _sample_map_values(points, map_data, origin, voxel_size):
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float32)

    grid_points = (np.asarray(points, dtype=np.float32) - origin[None, :]) / voxel_size[None, :]
    coords = np.stack(
        [grid_points[:, 2], grid_points[:, 1], grid_points[:, 0]],
        axis=0,
    ).astype(np.float32, copy=False)
    sampled = map_coordinates(
        map_data,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.asarray(sampled, dtype=np.float32)


def _score_protein_chains(protein_records, map_data, origin, voxel_size):
    print("Compute protein-chain scores against CA map")
    for chain_record in protein_records:
        ca_pos = _ca_positions(chain_record)
        sampled = _sample_map_values(ca_pos, map_data, origin, voxel_size)
        chain_record.score = float(np.sum(sampled))
        print(
            "  protein chain {} score = {:.4f}".format(
                chain_record.global_index,
                chain_record.score,
            )
        )


def _compute_clash_matrix(protein_records, clash_threshold, clash_distance, clash_resolution):
    n = len(protein_records)
    clash_matrix = np.zeros((n, n), dtype=np.int32)
    pair_summaries = []
    print("Compute protein-protein clash matrix")
    for i in range(n):
        pos_i = _ca_positions(protein_records[i])
        for j in range(i + 1, n):
            pos_j = _ca_positions(protein_records[j])
            if len(pos_i) == 0 or len(pos_j) == 0:
                clash_ratio_i = 0.0
                clash_ratio_j = 0.0
            else:
                clash_i, clash_j = get_clash(
                    pos_i,
                    np.ones(len(pos_i), dtype=np.float32),
                    pos_j,
                    np.ones(len(pos_j), dtype=np.float32),
                    resol=clash_resolution,
                    clash_dist=clash_distance,
                )
                clash_ratio_i = float(np.sum(clash_i) / max(len(pos_i), 1))
                clash_ratio_j = float(np.sum(clash_j) / max(len(pos_j), 1))

            incompatible = clash_ratio_i > clash_threshold or clash_ratio_j > clash_threshold
            if incompatible:
                clash_matrix[i, j] = 1
                clash_matrix[j, i] = 1
                protein_records[i].clash_partners.append(protein_records[j].global_index)
                protein_records[j].clash_partners.append(protein_records[i].global_index)

            pair_summaries.append(
                {
                    "chain_i": protein_records[i].global_index,
                    "chain_j": protein_records[j].global_index,
                    "clash_ratio_i": clash_ratio_i,
                    "clash_ratio_j": clash_ratio_j,
                    "incompatible": bool(incompatible),
                }
            )
    return clash_matrix, pair_summaries


def _scaled_scores(scores):
    return [int(round(float(score) * SCORE_SCALE)) for score in scores]


def _build_cp_sat_model(weights, clash_matrix, *, fixed_weight=None, fixed_size=None, maximize_size=False, minimize_lex=False):
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    variables = [model.NewBoolVar(f"chain_{idx}") for idx in range(len(weights))]

    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            if clash_matrix[i, j]:
                model.Add(variables[i] + variables[j] <= 1)

    objective_expr = sum(weight * var for weight, var in zip(weights, variables, strict=True))
    size_expr = sum(variables)
    lex_expr = sum((idx + 1) * variables[idx] for idx in range(len(variables)))

    if fixed_weight is not None:
        model.Add(objective_expr == fixed_weight)
    if fixed_size is not None:
        model.Add(size_expr == fixed_size)

    if maximize_size:
        model.Maximize(size_expr)
    elif minimize_lex:
        model.Minimize(lex_expr)
    else:
        model.Maximize(objective_expr)

    return model, variables, objective_expr, size_expr


def _solve_max_weight_independent_set(scores, clash_matrix, time_limit, num_workers, log_search_progress):
    from ortools.sat.python import cp_model

    if len(scores) == 0:
        return [], 0.0, "EMPTY"

    weights = _scaled_scores(scores)

    model1, vars1, obj1, _size1 = _build_cp_sat_model(weights, clash_matrix)
    solver1 = cp_model.CpSolver()
    solver1.parameters.max_time_in_seconds = float(time_limit)
    solver1.parameters.num_search_workers = int(num_workers)
    solver1.parameters.random_seed = 0
    solver1.parameters.log_search_progress = bool(log_search_progress)
    status1 = solver1.Solve(model1)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools failed to solve the chain assembly problem, status={status1}")

    best_weight = int(round(solver1.Value(obj1)))

    model2, vars2, _obj2, size2 = _build_cp_sat_model(
        weights,
        clash_matrix,
        fixed_weight=best_weight,
        maximize_size=True,
    )
    solver2 = cp_model.CpSolver()
    solver2.parameters.max_time_in_seconds = float(time_limit)
    solver2.parameters.num_search_workers = int(num_workers)
    solver2.parameters.random_seed = 0
    solver2.parameters.log_search_progress = False
    status2 = solver2.Solve(model2)
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [idx for idx, var in enumerate(vars1) if solver1.BooleanValue(var)]
        return selected, best_weight / SCORE_SCALE, cp_model.CpSolver().StatusName(status1)

    best_size = int(round(solver2.Value(size2)))

    model3, vars3, _obj3, _size3 = _build_cp_sat_model(
        weights,
        clash_matrix,
        fixed_weight=best_weight,
        fixed_size=best_size,
        minimize_lex=True,
    )
    solver3 = cp_model.CpSolver()
    solver3.parameters.max_time_in_seconds = float(time_limit)
    solver3.parameters.num_search_workers = int(num_workers)
    solver3.parameters.random_seed = 0
    solver3.parameters.log_search_progress = False
    status3 = solver3.Solve(model3)
    if status3 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [idx for idx, var in enumerate(vars2) if solver2.BooleanValue(var)]
        return selected, best_weight / SCORE_SCALE, cp_model.CpSolver().StatusName(status2)

    selected = [idx for idx, var in enumerate(vars3) if solver3.BooleanValue(var)]
    return selected, best_weight / SCORE_SCALE, cp_model.CpSolver().StatusName(status3)


def _resolve_output_paths(output_value):
    output_value = abspath(output_value)
    suffix = os.path.splitext(output_value)[1].lower()
    if suffix in {".cif", ".pdb"}:
        output_dir = os.path.dirname(output_value)
        output_cif = output_value
    else:
        output_dir = output_value
        output_cif = pjoin(output_dir, "assemble.cif")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_cif


def _write_selected_chains(output_cif, selected_records):
    if not selected_records:
        raise ValueError("No chains were selected for output.")

    chains_atom_pos_to_pdb(
        output_cif,
        chains_atom_pos=[record.atom_pos for record in selected_records],
        chains_atom_mask=[record.atom_mask for record in selected_records],
        chains_res_type=[record.res_type for record in selected_records],
        chains_res_idx=[record.res_idx for record in selected_records],
        chains_bfactor=[record.bfactor for record in selected_records],
        suffix="cif",
    )


def _record_to_summary(chain_record):
    return {
        "global_index": chain_record.global_index,
        "input_path": chain_record.input_path,
        "input_file_index": chain_record.input_file_index,
        "source_chain_index": chain_record.source_chain_index,
        "chain_type": chain_record.chain_type,
        "residue_count": chain_record.residue_count,
        "ca_count": chain_record.ca_count,
        "score": chain_record.score,
        "keep": chain_record.keep,
        "keep_reason": chain_record.keep_reason,
        "clash_partners": sorted(chain_record.clash_partners),
    }


def run_chain_assemble(
    structure_paths,
    ca_map_path,
    output,
    *,
    clash_threshold=0.10,
    clash_distance=1.0,
    clash_resolution=5.0,
    map_percentile=99.9,
    time_limit=120.0,
    num_workers=8,
    log_search_progress=False,
):
    output_dir, output_cif = _resolve_output_paths(output)
    structure_paths = _resolve_structure_paths(structure_paths)
    chain_records = _load_chain_records(structure_paths)

    raw_map_data, origin, voxel_size = _read_ca_map(abspath(ca_map_path))
    map_data = _normalize_density_map(raw_map_data, percentile=map_percentile)
    print(f"Read CA map from {abspath(ca_map_path)}")
    print(f"Map voxel size = {[float(x) for x in voxel_size]}")

    protein_records = [record for record in chain_records if record.chain_type == "protein"]
    retained_records = []
    for record in chain_records:
        if record.chain_type == "nucleic":
            record.keep = True
            record.keep_reason = "always_keep_nucleic"
            retained_records.append(record)
        elif record.chain_type in {"mixed", "other"}:
            record.keep = True
            record.keep_reason = "always_keep_nonprotein"
            retained_records.append(record)

    pair_summaries = []
    solver_status = "SKIPPED"
    selected_local_indices = []
    selected_protein_records = []
    selected_score = 0.0

    if protein_records:
        _score_protein_chains(protein_records, map_data, origin, voxel_size)
        clash_matrix, pair_summaries = _compute_clash_matrix(
            protein_records,
            clash_threshold=clash_threshold,
            clash_distance=clash_distance,
            clash_resolution=clash_resolution,
        )
        selected_local_indices, selected_score, solver_status = _solve_max_weight_independent_set(
            [record.score if record.score is not None else 0.0 for record in protein_records],
            clash_matrix,
            time_limit=time_limit,
            num_workers=num_workers,
            log_search_progress=log_search_progress,
        )
        selected_local_index_set = set(selected_local_indices)
        for local_index, record in enumerate(protein_records):
            if local_index in selected_local_index_set:
                record.keep = True
                record.keep_reason = "selected_by_solver"
                selected_protein_records.append(record)
                retained_records.append(record)
            else:
                record.keep = False
                record.keep_reason = "filtered_by_solver"
        print(
            "Selected {} / {} protein chains, objective score = {:.4f}, solver status = {}".format(
                len(selected_protein_records),
                len(protein_records),
                selected_score,
                solver_status,
            )
        )
    else:
        print("No protein chains found, output retained non-protein chains only")

    retained_records = [record for record in chain_records if record.keep]
    _write_selected_chains(output_cif, retained_records)
    print(f"Write assembled chains to {output_cif}")

    summary = {
        "input_structures": structure_paths,
        "ca_map_path": abspath(ca_map_path),
        "output_cif": output_cif,
        "clash_threshold": clash_threshold,
        "clash_distance": clash_distance,
        "clash_resolution": clash_resolution,
        "map_percentile": map_percentile,
        "solver_status": solver_status,
        "selected_protein_chain_indices": [protein_records[idx].global_index for idx in selected_local_indices],
        "selected_protein_score": selected_score,
        "chains": [_record_to_summary(record) for record in chain_records],
        "protein_pair_clashes": pair_summaries,
    }
    summary_path = pjoin(output_dir, "assemble_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Write assembly summary to {summary_path}")
    return summary


def add_args(parser):
    parser.add_argument(
        "--pdb",
        "--structure",
        "-p",
        dest="structure_paths",
        nargs="+",
        required=True,
        help="Input structure file(s) or directories containing .pdb/.cif files",
    )
    parser.add_argument(
        "--map",
        "--ca-map",
        "-m",
        dest="ca_map_path",
        required=True,
        help="Input CA density map (e.g. ca.mrc)",
    )
    parser.add_argument(
        "--out",
        "--output",
        "-o",
        dest="output",
        required=True,
        help="Output directory or output .cif path",
    )
    parser.add_argument(
        "--clash-threshold",
        type=float,
        default=0.10,
        help="If either chain's clash ratio exceeds this threshold, the pair is incompatible",
    )
    parser.add_argument(
        "--clash-distance",
        type=float,
        default=1.0,
        help="CA-CA clash distance passed to the EMProt-style clash kernel",
    )
    parser.add_argument(
        "--clash-resolution",
        type=float,
        default=5.0,
        help="Resolution parameter passed to the clash kernel",
    )
    parser.add_argument(
        "--map-percentile",
        type=float,
        default=99.9,
        help="Upper percentile used to clip and normalize the input CA map before scoring",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="OR-Tools solve time limit in seconds",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of OR-Tools workers",
    )
    parser.add_argument(
        "--log-search-progress",
        action="store_true",
        help="Enable OR-Tools search logging",
    )
    return parser


def main(args):
    return run_chain_assemble(
        structure_paths=args.structure_paths,
        ca_map_path=args.ca_map_path,
        output=args.output,
        clash_threshold=args.clash_threshold,
        clash_distance=args.clash_distance,
        clash_resolution=args.clash_resolution,
        map_percentile=args.map_percentile,
        time_limit=args.time_limit,
        num_workers=args.num_workers,
        log_search_progress=args.log_search_progress,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    add_args(parser)
    main(parser.parse_args())
