from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math

import numpy as np
from scipy.spatial import cKDTree

from em3dfit.config import Params
from em3dfit.flexible import RefinedPoseBundle, generate_refined_pose_bundles, refine_chain_segments
from em3dfit.graph_search import build_compatibility_graph, covered_chain_count, select_best_clique
from em3dfit.rigid import Pose, build_search_context, search_chain_poses
from em3dfit.score import transform_points
from em3dfit.types import Chain, linear_segment_adjacency
from em3dfit.utils import log_message as _base_log_message, stage_timer as _base_stage_timer

ASSEMBLY_MAX_CANDIDATES_PER_CHAIN = 30
LOG_STAGE = "Assemble"
log_message = partial(_base_log_message, stage=LOG_STAGE)
stage_timer = partial(_base_stage_timer, stage=LOG_STAGE)


def _ensure_segment_adjacency(chain: Chain) -> None:
    expected_shape = (chain.n_segments, chain.n_segments)
    if chain.segment_adjacency.shape != expected_shape:
        if chain.segment_links:
            adjacency = np.zeros(expected_shape, dtype=np.bool_)
            for link in chain.segment_links:
                left = int(link.left_segment) - 1
                right = int(link.right_segment) - 1
                if 0 <= left < chain.n_segments and 0 <= right < chain.n_segments and left != right:
                    adjacency[left, right] = True
                    adjacency[right, left] = True
            chain.segment_adjacency = adjacency
            return
        chain.segment_adjacency = linear_segment_adjacency(chain.n_segments)


def auto_define_domains(chain: Chain, target_residues: int) -> None:
    if chain.n_residues <= target_residues:
        _ensure_segment_adjacency(chain)
        return
    if chain.n_segments > 1:
        _ensure_segment_adjacency(chain)
        return

    unique_residues = np.unique(chain.residue_numbers)
    residues_per_domain = max(1, target_residues)
    residue_to_segment: dict[int, int] = {}
    for idx, residue in enumerate(unique_residues):
        residue_to_segment[int(residue)] = min(
            math.ceil((idx + 1) / residues_per_domain),
            math.ceil(len(unique_residues) / residues_per_domain),
        )

    new_segments = np.asarray([residue_to_segment[int(r)] for r in chain.residue_numbers], dtype=np.int32)
    chain.segment_numbers = new_segments
    chain.n_segments = int(new_segments.max(initial=1))
    chain.segment_adjacency = linear_segment_adjacency(chain.n_segments)
    chain.segment_links = []
    chain.frag_numbers = new_segments.copy()
    chain.n_frags = chain.n_segments


@dataclass(slots=True)
class PoseChoice:
    pose: Pose
    coords: np.ndarray
    score: float
    external_clash: float
    segment_solutions: np.ndarray | None = None
    external_clash_max: float = 0.0


@dataclass(slots=True)
class CycleSettings:
    pair_clash_cutoff: float
    external_clash_pair_cutoff: float
    external_clash_cutoff: float
    ldp_prune_exponent: float


@dataclass(slots=True)
class CycleMetrics:
    cycle: int
    chain_count_before: int
    selected_chains: int
    graph_vertices: int
    graph_edges: int
    clique_size: int
    ldps_before: int
    ldps_after: int


def transform_chain_coords(chain: Chain, solution: np.ndarray) -> np.ndarray:
    return transform_points(chain.coords, chain.centroid, solution).astype(np.float32, copy=False)


def _assembled_chain_clash_scores(
    coords: np.ndarray,
    assembled_coords: list[np.ndarray],
    params: Params,
) -> np.ndarray:
    if not assembled_coords:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(
        [_pairwise_clash_score(coords, existing_coords, params) for existing_coords in assembled_coords],
        dtype=np.float32,
    )



def _pairwise_clash_score(coords_a: np.ndarray, coords_b: np.ndarray, params: Params) -> float:
    if coords_a.size == 0 or coords_b.size == 0:
        return 0.0
    rsigma2 = (np.pi / (2.4 + 0.8 * params.resol)) ** 2
    bw = np.sqrt(6.0 / rsigma2)
    tree = cKDTree(coords_b)
    dists, _ = tree.query(coords_a, distance_upper_bound=bw)
    valid = np.isfinite(dists)
    if not np.any(valid):
        return 0.0
    scores = np.zeros(len(coords_a), dtype=np.float32)
    d2s = np.maximum(dists[valid] - params.clash_dist, 0.0) ** 2
    scores[valid] = np.exp(-rsigma2 * d2s).astype(np.float32, copy=False)
    return float(np.mean(scores))


def _occupancy_scores(ldps: np.ndarray, coords: np.ndarray, params: Params) -> np.ndarray:
    if ldps.size == 0 or coords.size == 0:
        return np.zeros(len(ldps), dtype=np.float32)
    rsigma2 = (np.pi / (2.4 + 0.8 * params.resol)) ** 2
    bw = np.sqrt(6.0 / rsigma2)
    tree = cKDTree(coords)
    dists, _ = tree.query(ldps, distance_upper_bound=bw)
    valid = np.isfinite(dists)
    scores = np.zeros(len(ldps), dtype=np.float32)
    if np.any(valid):
        d2s = np.maximum(dists[valid] - params.clash_dist, 0.0) ** 2
        scores[valid] = np.exp(-rsigma2 * d2s).astype(np.float32, copy=False)
    return scores


def _precompute_pairwise_clashes(
    local_indices: list[int],
    choices: list[list[PoseChoice]],
    params: Params,
) -> dict[tuple[int, int, int, int], float]:
    pair_clash: dict[tuple[int, int, int, int], float] = {}
    with stage_timer("pairwise pose clash matrix"):
        for i in range(len(local_indices) - 1):
            for j in range(i + 1, len(local_indices)):
                for pi, pose_i in enumerate(choices[i]):
                    for pj, pose_j in enumerate(choices[j]):
                        score_ij = _pairwise_clash_score(pose_i.coords, pose_j.coords, params)
                        score_ji = _pairwise_clash_score(pose_j.coords, pose_i.coords, params)
                        pair_clash[(i, pi, j, pj)] = score_ij
                        pair_clash[(j, pj, i, pi)] = score_ji
    return pair_clash


def _directional_pair_clash(
    pair_clash: dict[tuple[int, int, int, int], float],
    chain_i: int,
    pose_i: int,
    chain_j: int,
    pose_j: int,
) -> tuple[float, float]:
    score_ij = float(pair_clash[(chain_i, pose_i, chain_j, pose_j)])
    score_ji = float(pair_clash[(chain_j, pose_j, chain_i, pose_i)])
    return score_ij, score_ji


def _clique_chain_diagnostics(
    limited_choices: list[list[PoseChoice]],
    selection: list[int],
    graph,
    pair_clash: dict[tuple[int, int, int, int], float],
) -> tuple[list[str], list[str]]:
    vertex_indices = {
        (vertex.chain_idx, vertex.pose_idx): vertex.index
        for vertex in graph.vertices
    }
    selected_lines: list[str] = []
    unselected_lines: list[str] = []

    for chain_idx, pose_idx in enumerate(selection):
        if pose_idx < 0:
            continue
        choice = limited_choices[chain_idx][pose_idx]
        vertex_idx = vertex_indices.get((chain_idx, pose_idx))
        neighbor_count = len(graph.adjacency[vertex_idx]) if vertex_idx is not None else 0
        selected_lines.append(
            f"chain={chain_idx} pose={pose_idx} score={choice.score:.3f} "
            f"ext_sum={choice.external_clash:.3f} ext_max={choice.external_clash_max:.3f} "
            f"neighbors={neighbor_count}"
        )

    for chain_idx, chain_choices in enumerate(limited_choices):
        if selection[chain_idx] >= 0 or not chain_choices:
            continue
        pose_idx = 0
        choice = chain_choices[pose_idx]
        vertex_idx = vertex_indices.get((chain_idx, pose_idx))
        blocking_selected = 0
        max_block_clash = 0.0
        compatible_selected = 0
        for other_chain_idx, other_pose_idx in enumerate(selection):
            if other_pose_idx < 0:
                continue
            other_vertex_idx = vertex_indices.get((other_chain_idx, other_pose_idx))
            if vertex_idx is not None and other_vertex_idx is not None and other_vertex_idx in graph.adjacency[vertex_idx]:
                compatible_selected += 1
            else:
                blocking_selected += 1
                if (chain_idx, pose_idx, other_chain_idx, other_pose_idx) in pair_clash:
                    score_ij, score_ji = _directional_pair_clash(pair_clash, chain_idx, pose_idx, other_chain_idx, other_pose_idx)
                    max_block_clash = max(max_block_clash, score_ij, score_ji)
        unselected_lines.append(
            f"chain={chain_idx} top_pose={pose_idx} score={choice.score:.3f} "
            f"ext_sum={choice.external_clash:.3f} ext_max={choice.external_clash_max:.3f} "
            f"compatible_selected={compatible_selected} blocking_selected={blocking_selected} "
            f"max_block_clash={max_block_clash:.3f}"
        )

    return selected_lines, unselected_lines


def _refine_and_place_chain(
    chain: Chain,
    selected_choice: PoseChoice,
    ldps: np.ndarray,
    ldps_dens: np.ndarray,
    assembled_coords: list[np.ndarray],
    params: Params,
) -> np.ndarray:
    if selected_choice.segment_solutions is not None:
        chain.solutions = selected_choice.segment_solutions.astype(np.float32, copy=True)
        coords = selected_choice.coords.astype(np.float32, copy=True)
    elif params.flexible and chain.n_segments > 1:
        context = build_search_context(ldps, ldps_dens, chain, params)
        merged = np.concatenate(assembled_coords, axis=0) if assembled_coords else np.empty((0, 3), dtype=np.float32)
        chain.solutions = refine_chain_segments(
            chain,
            selected_choice.pose.solution,
            context,
            merged,
            params,
            base_pose_score=selected_choice.pose.score,
        )
        coords = np.concatenate(
            [
                transform_points(chain.coords[chain.segment_numbers == segment], chain.centroid, chain.solutions[segment - 1])
                for segment in range(1, chain.n_segments + 1)
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    else:
        chain.solutions = np.tile(selected_choice.pose.solution.astype(np.float32, copy=False), (chain.n_segments, 1))
        coords = transform_chain_coords(chain, selected_choice.pose.solution)
    return coords


def _prune_ldps(
    ldps: np.ndarray,
    ldps_dens: np.ndarray,
    placed_coords: list[np.ndarray],
    params: Params,
    cycle_settings: CycleSettings,
) -> tuple[np.ndarray, np.ndarray]:
    if not placed_coords:
        return ldps, ldps_dens
    merged = np.concatenate(placed_coords, axis=0)
    occ = _occupancy_scores(ldps, merged, params)
    suppression = np.power(np.clip(1.0 - occ, 0.0, 1.0), cycle_settings.ldp_prune_exponent).astype(np.float32, copy=False)
    new_dens = ldps_dens * suppression
    keep_floor = max(1e-4, float(np.max(ldps_dens)) * 1e-3)
    keep = new_dens > keep_floor
    min_keep = max(1, int(len(ldps) * params.ldp_min_keep_ratio))
    if int(np.count_nonzero(keep)) < min_keep:
        top_ids = np.argsort(new_dens)[-min_keep:]
        keep = np.zeros(len(ldps), dtype=bool)
        keep[top_ids] = True
    return ldps[keep], new_dens[keep]


def _build_simple_round_choices(
    chain: Chain,
    poses: list[Pose],
    refined_bundles: list[RefinedPoseBundle],
    assembled_coords: list[np.ndarray],
    params: Params,
    max_candidates: int = ASSEMBLY_MAX_CANDIDATES_PER_CHAIN,
) -> list[PoseChoice]:
    choices: list[PoseChoice] = []
    if refined_bundles:
        for bundle in refined_bundles:
            clash_scores = _assembled_chain_clash_scores(bundle.coords, assembled_coords, params)
            external_clash = float(np.sum(clash_scores, dtype=np.float32)) if clash_scores.size else 0.0
            external_clash_max = float(np.max(clash_scores)) if clash_scores.size else 0.0
            if external_clash_max > params.external_clash_pair_cutoff or external_clash > params.external_clash_cutoff:
                continue
            choices.append(
                PoseChoice(
                    pose=bundle.pose,
                    coords=bundle.coords.astype(np.float32, copy=False),
                    score=float(bundle.score),
                    external_clash=external_clash,
                    segment_solutions=bundle.segment_solutions.astype(np.float32, copy=False),
                    external_clash_max=external_clash_max,
                )
            )
    else:
        for pose in poses:
            coords = transform_chain_coords(chain, pose.solution)
            clash_scores = _assembled_chain_clash_scores(coords, assembled_coords, params)
            external_clash = float(np.sum(clash_scores, dtype=np.float32)) if clash_scores.size else 0.0
            external_clash_max = float(np.max(clash_scores)) if clash_scores.size else 0.0
            if external_clash_max > params.external_clash_pair_cutoff or external_clash > params.external_clash_cutoff:
                continue
            choices.append(
                PoseChoice(
                    pose=pose,
                    coords=coords,
                    score=float(pose.score),
                    external_clash=external_clash,
                    external_clash_max=external_clash_max,
                )
            )

    choices.sort(
        key=lambda choice: (
            choice.score,
            choice.external_clash_max,
            choice.external_clash,
        )
    )
    return choices[: max(max_candidates, 1)]


def _select_pose_subset_via_graph_simple(
    choices: list[list[PoseChoice]],
    params: Params,
) -> tuple[list[int], int, int, int]:
    pair_clash = _precompute_pairwise_clashes(list(range(len(choices))), choices, params)
    pose_counts = [len(chain_choices) for chain_choices in choices]
    pose_scores = [[float(choice.score) for choice in chain_choices] for chain_choices in choices]

    def clash_lookup(chain_i: int, pose_i: int, chain_j: int, pose_j: int) -> float:
        score_ij, score_ji = _directional_pair_clash(pair_clash, chain_i, pose_i, chain_j, pose_j)
        return max(score_ij, score_ji)

    with stage_timer("compatibility graph build"):
        graph = build_compatibility_graph(
            pose_counts=pose_counts,
            pose_scores=pose_scores,
            pair_clash_lookup=clash_lookup,
            clash_cutoff=params.clash_cutoff,
            max_vertices=max(pose_counts, default=0),
        )

    log_message(
        f"compatibility graph: {len(graph.vertices)} vertices, {graph.edge_count} edges, "
        f"pose_counts={pose_counts}, clash_cutoff={params.clash_cutoff:.3f}"
    )

    selection, stats = select_best_clique(graph)
    if selection is None:
        return [-1 for _ in choices], len(graph.vertices), graph.edge_count, 0

    covered = covered_chain_count(selection)
    log_message(
        f"selected clique score total={stats.best_score:.3f}, "
        f"vertex={stats.best_vertex_score:.3f}, edge_penalty={stats.best_edge_penalty:.3f}"
    )
    selected_lines, unselected_lines = _clique_chain_diagnostics(choices, selection, graph, pair_clash)
    if selected_lines:
        log_message("selected chains " + " | ".join(selected_lines))
    if unselected_lines:
        log_message("unselected chains " + " | ".join(unselected_lines))
    if covered < len(choices):
        log_message(f"graph search selected a partial clique covering {covered}/{len(choices)} chains")
    return selection, len(graph.vertices), graph.edge_count, stats.best_size


def assemble_chains(chains: list[Chain], ldps: np.ndarray, ldps_dens: np.ndarray, params: Params) -> list[list[Pose]]:
    with stage_timer("auto domain definition"):
        for chain in chains:
            auto_define_domains(chain, params.auto_domain_residues)

    for chain in chains:
        chain.solutions = None

    all_pose_sets: list[list[Pose]] = [[] for _ in chains]
    available_ldps = ldps.astype(np.float32, copy=True)
    available_dens = ldps_dens.astype(np.float32, copy=True)
    assembled_coords: list[np.ndarray] = []
    placed_globals: set[int] = set()

    for cycle in range(1, max(params.assembly_cycles, 1) + 1):
        remaining = [idx for idx in range(len(chains)) if idx not in placed_globals]
        if not remaining:
            break

        log_message(
            f"assembly cycle {cycle}: {len(remaining)} chain(s) remaining, "
            f"{len(available_ldps)} ldps available"
        )

        local_pose_sets: list[list[Pose]] = []
        local_refined_bundles: list[list[RefinedPoseBundle]] = []
        local_choices: list[list[PoseChoice]] = []
        merged = np.concatenate(assembled_coords, axis=0) if assembled_coords else np.empty((0, 3), dtype=np.float32)

        with stage_timer(f"cycle {cycle} rigid pose generation"):
            for global_idx in remaining:
                poses = search_chain_poses(
                    chains[global_idx],
                    available_ldps,
                    available_dens,
                    params,
                    rigid_score_cutoff=params.rigid_cutoff_score_late,
                    rigid_nleast=params.rigid_nleast,
                )
                all_pose_sets[global_idx] = poses
                local_pose_sets.append(poses)

        with stage_timer(f"cycle {cycle} flexible candidate generation"):
            for global_idx, poses in zip(remaining, local_pose_sets, strict=True):
                chain = chains[global_idx]
                bundles: list[RefinedPoseBundle] = []
                if params.flexible and chain.n_segments > 1 and poses:
                    context = build_search_context(available_ldps, available_dens, chain, params)
                    bundles = generate_refined_pose_bundles(
                        chain,
                        poses,
                        context,
                        merged,
                        params,
                        max_bundles=max(ASSEMBLY_MAX_CANDIDATES_PER_CHAIN, params.flexible_bundle_limit),
                    )
                local_refined_bundles.append(bundles)

        for global_idx, poses, refined_bundles in zip(remaining, local_pose_sets, local_refined_bundles, strict=True):
            choices = _build_simple_round_choices(
                chains[global_idx],
                poses,
                refined_bundles,
                assembled_coords,
                params,
                max_candidates=ASSEMBLY_MAX_CANDIDATES_PER_CHAIN,
            )
            local_choices.append(choices)
            log_message(
                f"cycle {cycle} chain {chains[global_idx].index:02d} kept "
                f"{len(choices)} candidate placement(s)"
            )

        if not any(local_choices):
            log_message("no valid candidates remain after clash filtering against placed chains; stopping assembly")
            break

        selected_local_ids, graph_vertices, graph_edges, clique_size = _select_pose_subset_via_graph_simple(
            local_choices,
            params,
        )
        selected_local_indices = [idx for idx, pose_idx in enumerate(selected_local_ids) if pose_idx >= 0]

        if not selected_local_indices:
            log_message("no chains were selected in this cycle; stopping assembly")
            break

        newly_placed_coords: list[np.ndarray] = []
        with stage_timer(f"cycle {cycle} final refinement"):
            for local_idx in selected_local_indices:
                global_idx = remaining[local_idx]
                chain = chains[global_idx]
                selected_choice = local_choices[local_idx][selected_local_ids[local_idx]]
                coords = _refine_and_place_chain(chain, selected_choice, available_ldps, available_dens, assembled_coords, params)
                assembled_coords.append(coords)
                newly_placed_coords.append(coords)
                placed_globals.add(global_idx)

        with stage_timer(f"cycle {cycle} ldp weakening"):
            before = len(available_ldps)
            available_ldps, available_dens = _prune_ldps(
                available_ldps,
                available_dens,
                newly_placed_coords,
                params,
                CycleSettings(
                    pair_clash_cutoff=params.clash_cutoff,
                    external_clash_pair_cutoff=params.external_clash_pair_cutoff,
                    external_clash_cutoff=params.external_clash_cutoff,
                    ldp_prune_exponent=params.ldp_prune_exponent,
                ),
            )
        previous_metrics = CycleMetrics(
            cycle=cycle,
            chain_count_before=len(remaining),
            selected_chains=len(selected_local_indices),
            graph_vertices=graph_vertices,
            graph_edges=graph_edges,
            clique_size=clique_size,
            ldps_before=before,
            ldps_after=len(available_ldps),
        )
        log_message(
            f"cycle {cycle} result: selected={previous_metrics.selected_chains}/{previous_metrics.chain_count_before}, "
            f"clique_size={previous_metrics.clique_size}, graph_edges={previous_metrics.graph_edges}, "
            f"ldps={previous_metrics.ldps_after}/{previous_metrics.ldps_before}"
        )

    remaining = [idx for idx in range(len(chains)) if idx not in placed_globals]
    if remaining:
        log_message(
            f"assembly ended with {len(remaining)} unresolved chain(s); "
            f"only placed chains will be written: {remaining}"
        )

    return all_pose_sets
