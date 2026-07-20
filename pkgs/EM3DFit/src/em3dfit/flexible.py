from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy import ndimage as ndi, optimize
from scipy.spatial import cKDTree

from em3dfit.config import Params
from em3dfit.rigid import Pose
from em3dfit.rigid import SearchContext, _build_smoothed_grid
from em3dfit.score import clash_score, transform_points
from em3dfit.types import Chain, linear_segment_adjacency, segment_anchor_pairs
from em3dfit.utils import log_message as _base_log_message, stage_timer as _base_stage_timer

LOG_STAGE = "Flexible"
log_message = partial(_base_log_message, stage=LOG_STAGE)
stage_timer = partial(_base_stage_timer, stage=LOG_STAGE)

ProgressFeedbackKey = int | tuple[int, int] | tuple[int, int, tuple[int, ...]]
ProgressFeedbackBias = float | dict[ProgressFeedbackKey, float]
PROGRESS_FEEDBACK_BRANCH_DEPTH = 3


@dataclass(slots=True)
class SegmentRestraint:
    local_anchor: np.ndarray
    neighbor_anchor: np.ndarray
    target_delta: np.ndarray
    neighbor_segment: int


@dataclass(slots=True)
class RefinedPoseBundle:
    pose: Pose
    pose_rank: int
    seed_segment: int
    growth_order: tuple[int, ...]
    segment_solutions: np.ndarray
    coords: np.ndarray
    score: float
    grid_score: float | None = None
    segment_scores: np.ndarray | None = None
    normalized_score: float | None = None
    progress_stage_count: int = 0
    progress_worst_normalized_score: float | None = None
    progress_weighted_normalized_score: float | None = None
    progress_feedback_penalty: float = 0.0
    progress_stage_scores: tuple[float, ...] = ()
    branch_family: tuple[int, ...] = ()


@dataclass(slots=True)
class GrowthStageSnapshot:
    growth_order: tuple[int, ...]
    segment_solutions: np.ndarray
    child_normalized_score: float | None = None
    pair_normalized_score: float | None = None


@dataclass(slots=True)
class SeedScreenCandidate:
    pose: Pose
    pose_rank: int
    seed_segment: int
    growth_order: list[int]
    growth_parents: dict[int, int]
    score: float
    normalized_score: float | None = None
    seed_solution: np.ndarray | None = None


@dataclass(slots=True)
class SegmentScoreStats:
    mean: float
    std: float


def growth_branch_family(
    growth_order: tuple[int, ...] | list[int],
    max_depth: int = PROGRESS_FEEDBACK_BRANCH_DEPTH,
) -> tuple[int, ...]:
    normalized_order = tuple(int(segment) for segment in growth_order if int(segment) > 0)
    if max_depth <= 0:
        return ()
    return normalized_order[:max_depth]


def progress_feedback_bucket(
    growth_order: tuple[int, ...] | list[int],
    seed_segment: int,
    branch_family: tuple[int, ...] | None = None,
) -> tuple[int, int, tuple[int, ...]]:
    normalized_order = tuple(int(segment) for segment in growth_order if int(segment) > 0)
    normalized_seed = max(int(seed_segment), 0)
    if not normalized_order or normalized_seed <= 0:
        return (0, 0, ())
    family = growth_branch_family(normalized_order) if branch_family is None else tuple(int(segment) for segment in branch_family)
    return (len(normalized_order), normalized_seed, family)


def _grid_score_world(
    points_world: np.ndarray,
    weights: np.ndarray,
    context: SearchContext,
    params: Params,
    refinement_grid: np.ndarray | None = None,
) -> float:
    centered = points_world - context.centrioda
    sample = ((centered - context.slowera) / params.sgrid).T
    grid = context.refinement_grid if refinement_grid is None else refinement_grid
    vals = ndi.map_coordinates(grid, sample, order=1, mode="constant", cval=0.0)
    return float(np.sum(vals * weights))


def _continuity_penalty(
    moved_anchor: np.ndarray,
    neighbor_anchor: np.ndarray,
    target_delta: np.ndarray,
    tolerance: float,
    weight: float,
) -> float:
    diff = moved_anchor - neighbor_anchor - target_delta
    distance = float(np.sqrt(np.dot(diff, diff)))
    excess = max(distance - tolerance, 0.0)
    return weight * excess * excess


def _segment_index_map(chain: Chain) -> dict[int, np.ndarray]:
    return {
        segment: np.flatnonzero(chain.segment_numbers == segment)
        for segment in range(1, chain.n_segments + 1)
    }


def _segment_orders(chain: Chain) -> list[list[int]]:
    size_order = sorted(
        range(1, chain.n_segments + 1),
        key=lambda segment: -int(np.count_nonzero(chain.segment_numbers == segment)),
    )
    adjacency = chain.segment_adjacency
    if adjacency.shape != (chain.n_segments, chain.n_segments):
        adjacency = linear_segment_adjacency(chain.n_segments)

    if not size_order:
        return [[]]

    graph_order: list[int] = []
    seen: set[int] = set()
    queue = [size_order[0]]
    while queue:
        segment = queue.pop(0)
        if segment in seen:
            continue
        seen.add(segment)
        graph_order.append(segment)
        neighbors = [
            neighbor + 1
            for neighbor, linked in enumerate(adjacency[segment - 1])
            if linked and (neighbor + 1) not in seen
        ]
        neighbors.sort(key=lambda candidate: -int(np.count_nonzero(chain.segment_numbers == candidate)))
        queue.extend(neighbors)

    for segment in size_order:
        if segment not in seen:
            graph_order.append(segment)

    return [size_order, graph_order[::-1]]


def _seed_growth_order(chain: Chain, seed_segment: int) -> list[int]:
    order, _parents = _seed_growth_tree(chain, seed_segment)
    return order


def _seed_growth_tree(chain: Chain, seed_segment: int) -> tuple[list[int], dict[int, int]]:
    adjacency = chain.segment_adjacency
    if adjacency.shape != (chain.n_segments, chain.n_segments):
        adjacency = linear_segment_adjacency(chain.n_segments)

    size_lookup = {
        segment: int(np.count_nonzero(chain.segment_numbers == segment))
        for segment in range(1, chain.n_segments + 1)
    }
    order: list[int] = []
    seen: set[int] = set()
    parents: dict[int, int] = {}
    queue = [seed_segment]
    while queue:
        segment = queue.pop(0)
        if segment in seen:
            continue
        seen.add(segment)
        order.append(segment)
        neighbors = [
            neighbor + 1
            for neighbor, linked in enumerate(adjacency[segment - 1])
            if linked and (neighbor + 1) not in seen
        ]
        neighbors.sort(key=lambda candidate: (-size_lookup[candidate], candidate))
        for neighbor in neighbors:
            parents.setdefault(neighbor, segment)
        queue.extend(neighbors)

    for segment in range(1, chain.n_segments + 1):
        if segment not in seen:
            order.append(segment)
    return order, parents


def _seed_segment_candidates(chain: Chain, limit: int) -> list[int]:
    if limit <= 0:
        return []
    segments = list(range(1, chain.n_segments + 1))
    segments.sort(
        key=lambda segment: (
            -int(np.count_nonzero(chain.segment_numbers == segment)),
            segment,
        )
    )
    return segments[: min(limit, len(segments))]


def _build_segment_restraints(
    chain: Chain,
    solutions: np.ndarray,
    base_solution: np.ndarray,
    segment_indices: dict[int, np.ndarray],
) -> dict[int, list[SegmentRestraint]]:
    base_world = transform_points(chain.coords, chain.centroid, base_solution)
    restraints: dict[int, list[SegmentRestraint]] = {segment: [] for segment in range(1, chain.n_segments + 1)}
    for left, right, left_anchor_idx, right_anchor_idx in segment_anchor_pairs(chain, segment_indices):
        left_anchor = chain.coords[left_anchor_idx : left_anchor_idx + 1]
        right_anchor = chain.coords[right_anchor_idx : right_anchor_idx + 1]
        left_neighbor_world = transform_points(right_anchor, chain.centroid, solutions[right - 1])[0]
        right_neighbor_world = transform_points(left_anchor, chain.centroid, solutions[left - 1])[0]
        left_target_delta = base_world[left_anchor_idx] - base_world[right_anchor_idx]
        right_target_delta = -left_target_delta

        restraints[left].append(
            SegmentRestraint(
                local_anchor=left_anchor[0].astype(np.float32, copy=False),
                neighbor_anchor=left_neighbor_world.astype(np.float32, copy=False),
                target_delta=left_target_delta.astype(np.float32, copy=False),
                neighbor_segment=right,
            )
        )
        restraints[right].append(
            SegmentRestraint(
                local_anchor=right_anchor[0].astype(np.float32, copy=False),
                neighbor_anchor=right_neighbor_world.astype(np.float32, copy=False),
                target_delta=right_target_delta.astype(np.float32, copy=False),
                neighbor_segment=left,
            )
        )

    return restraints


def _segment_objective(
    solution: np.ndarray,
    base_solution: np.ndarray,
    segment_points: np.ndarray,
    segment_weights: np.ndarray,
    chain_centroid: np.ndarray,
    context: SearchContext,
    fixed_coords: np.ndarray,
    grown_coords: np.ndarray,
    refinement_grid: np.ndarray,
    restraints: list[SegmentRestraint],
    params: Params,
) -> float:
    moved = transform_points(segment_points, chain_centroid, solution)
    score = _grid_score_world(moved, segment_weights, context, params, refinement_grid=refinement_grid)

    if fixed_coords.size:
        score += 50.0 * clash_score(moved, fixed_coords, params)
    if grown_coords.size:
        score += params.growth_occupancy_penalty_weight * _growth_occupancy_penalty(moved, grown_coords, params)

    if restraints:
        for restraint in restraints:
            moved_anchor = transform_points(
                restraint.local_anchor.reshape(1, 3),
                chain_centroid,
                solution,
            )[0]
            score += _continuity_penalty(
                moved_anchor=moved_anchor,
                neighbor_anchor=restraint.neighbor_anchor,
                target_delta=restraint.target_delta,
                tolerance=params.segment_link_tolerance,
                weight=params.segment_link_weight,
            )

    dt = solution[3:6] - base_solution[3:6]
    da = solution[:3] - base_solution[:3]
    angle_bound = math.radians(params.segment_angle_bound_deg)
    excess_t = np.maximum(np.abs(dt) - params.segment_shift_bound, 0.0)
    excess_a = np.maximum(np.abs(da) - angle_bound, 0.0)
    score += 0.25 * float(np.dot(dt, dt))
    score += 6.0 * float(np.dot(da, da))
    score += 4.0 * float(np.dot(excess_t, excess_t))
    score += 40.0 * float(np.dot(excess_a, excess_a))
    return float(score)


def _compose_segment_coords(chain: Chain, segment_solutions: np.ndarray) -> np.ndarray:
    coords = np.empty_like(chain.coords, dtype=np.float32)
    for segment in range(1, chain.n_segments + 1):
        ids = chain.segment_numbers == segment
        coords[ids] = transform_points(chain.coords[ids], chain.centroid, segment_solutions[segment - 1])
    return coords.astype(np.float32, copy=False)


def _segment_world_coords(
    chain: Chain,
    segment_indices: dict[int, np.ndarray],
    segment: int,
    segment_solutions: np.ndarray,
) -> np.ndarray:
    ids = segment_indices[segment]
    return transform_points(chain.coords[ids], chain.centroid, segment_solutions[segment - 1]).astype(np.float32, copy=False)


def _bundle_grid_score(
    chain: Chain,
    segment_solutions: np.ndarray,
    context: SearchContext,
    params: Params,
    refinement_grid: np.ndarray | None = None,
) -> float:
    coords = _compose_segment_coords(chain, segment_solutions)
    return _grid_score_world(coords, chain.weights, context, params, refinement_grid=refinement_grid)


def _bundle_segment_scores(
    chain: Chain,
    segment_solutions: np.ndarray,
    context: SearchContext,
    params: Params,
    refinement_grid: np.ndarray | None = None,
) -> np.ndarray:
    scores = np.empty(chain.n_segments, dtype=np.float32)
    segment_indices = _segment_index_map(chain)
    grid = context.refinement_grid if refinement_grid is None else refinement_grid
    for segment in range(1, chain.n_segments + 1):
        ids = segment_indices[segment]
        scores[segment - 1] = _score_segment_pose(
            chain,
            chain.coords[ids],
            chain.weights[ids],
            segment_solutions[segment - 1],
            context,
            params,
            grid,
        )
    return scores


def _bundle_total_score(segment_scores: np.ndarray) -> float:
    return float(np.sum(segment_scores, dtype=np.float64))


def _bundle_normalized_score(
    total_score: float,
    chain: Chain,
    params: Params,
) -> float | None:
    if chain.rigid_score_mean is None or chain.rigid_score_std is None:
        return None
    return float(
        (total_score - chain.rigid_score_mean) / max(chain.rigid_score_std, params.flexible_revert_min_std, 1e-6)
    )


def _score_refined_bundle(
    chain: Chain,
    segment_solutions: np.ndarray,
    context: SearchContext,
    params: Params,
) -> tuple[np.ndarray, float, float | None, float]:
    segment_scores = _bundle_segment_scores(chain, segment_solutions, context, params)
    total_score = _bundle_total_score(segment_scores)
    normalized_score = _bundle_normalized_score(total_score, chain, params)
    grid_score = _bundle_grid_score(chain, segment_solutions, context, params)
    return segment_scores, total_score, normalized_score, grid_score


def _score_segment_pose(
    chain: Chain,
    segment_points: np.ndarray,
    segment_weights: np.ndarray,
    solution: np.ndarray,
    context: SearchContext,
    params: Params,
    refinement_grid: np.ndarray,
) -> float:
    moved = transform_points(segment_points, chain.centroid, solution)
    return _grid_score_world(moved, segment_weights, context, params, refinement_grid=refinement_grid)


def _estimate_segment_score_stats(
    chain: Chain,
    poses: list[Pose],
    segment_indices: dict[int, np.ndarray],
    context: SearchContext,
    params: Params,
) -> dict[int, SegmentScoreStats]:
    if (
        chain.segment_score_means is not None
        and chain.segment_score_stds is not None
        and len(chain.segment_score_means) == chain.n_segments
        and len(chain.segment_score_stds) == chain.n_segments
    ):
        return {
            segment: SegmentScoreStats(
                mean=float(chain.segment_score_means[segment - 1]),
                std=max(float(chain.segment_score_stds[segment - 1]), params.flexible_revert_min_std),
            )
            for segment in range(1, chain.n_segments + 1)
        }

    if not poses:
        return {
            segment: SegmentScoreStats(mean=0.0, std=max(params.flexible_revert_min_std, 1e-3))
            for segment in range(1, chain.n_segments + 1)
        }

    stats: dict[int, SegmentScoreStats] = {}
    for segment in range(1, chain.n_segments + 1):
        ids = segment_indices[segment]
        segment_points = chain.coords[ids]
        segment_weights = chain.weights[ids]
        values = np.asarray(
            [
                _score_segment_pose(
                    chain,
                    segment_points,
                    segment_weights,
                    pose.solution,
                    context,
                    params,
                    context.refinement_grid,
                )
                for pose in poses
            ],
            dtype=np.float32,
        )
        mean = float(np.mean(values)) if values.size else 0.0
        std = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0
        stats[segment] = SegmentScoreStats(mean=mean, std=max(std, params.flexible_revert_min_std))
    return stats


def _segment_zscore(score: float, stats: SegmentScoreStats) -> float:
    return float((score - stats.mean) / max(stats.std, 1e-6))


def _should_revert_segment_refine(
    score: float,
    stats: SegmentScoreStats,
    params: Params,
) -> bool:
    return _segment_zscore(score, stats) > params.flexible_revert_zcut


def _should_revert_pair_refine(
    score: float,
    child_stats: SegmentScoreStats,
    parent_stats: SegmentScoreStats,
    params: Params,
) -> bool:
    child_z, parent_z = _pair_refine_zscores(score, child_stats, parent_stats)
    return child_z > params.flexible_revert_zcut and parent_z > params.flexible_revert_zcut


def _pair_refine_zscores(
    score: float,
    child_stats: SegmentScoreStats,
    parent_stats: SegmentScoreStats,
) -> tuple[float, float]:
    mean_sum = child_stats.mean + parent_stats.mean
    child_z = (score - mean_sum) / max(child_stats.std, 1e-6)
    parent_z = (score - mean_sum) / max(parent_stats.std, 1e-6)
    return float(child_z), float(parent_z)


def _pair_refine_normalized_score(
    score: float,
    child_stats: SegmentScoreStats,
    parent_stats: SegmentScoreStats,
) -> float:
    child_z, parent_z = _pair_refine_zscores(score, child_stats, parent_stats)
    return float(max(child_z, parent_z))


def _screen_seed_combination(
    chain: Chain,
    pose: Pose,
    seed_segment: int,
    context: SearchContext,
    assembled_coords: np.ndarray,
    params: Params,
    segment_stats: SegmentScoreStats | None = None,
) -> float:
    _solution, score = _refine_seed_segment(
        chain,
        pose,
        seed_segment,
        context,
        assembled_coords,
        params,
        segment_stats=segment_stats,
    )
    return score


def _refine_seed_segment(
    chain: Chain,
    pose: Pose,
    seed_segment: int,
    context: SearchContext,
    assembled_coords: np.ndarray,
    params: Params,
    segment_stats: SegmentScoreStats | None = None,
) -> tuple[np.ndarray, float]:
    seed_ids = chain.segment_numbers == seed_segment
    seed_points = chain.coords[seed_ids]
    seed_weights = chain.weights[seed_ids]
    fixed_coords = assembled_coords if assembled_coords.size else np.empty((0, 3), dtype=np.float32)
    x0 = pose.solution.astype(np.float32, copy=True)
    restrained_grid = _restrained_refinement_grid(
        context,
        transform_points(seed_points, chain.centroid, x0),
        context.refinement_grid,
        params,
        intensity=_growth_stage_intensity(
            chain.n_segments,
            {seed_segment},
            set(),
            params,
            rigid_score=pose.score,
            rigid_score_mean=chain.rigid_score_mean,
            rigid_score_std=chain.rigid_score_std,
        ),
    )
    score00 = _score_segment_pose(chain, seed_points, seed_weights, x0, context, params, restrained_grid)
    result = optimize.minimize(
        _segment_objective,
        x0=x0,
        args=(
            x0.copy(),
            seed_points,
            seed_weights,
            chain.centroid,
            context,
            fixed_coords,
            np.empty((0, 3), dtype=np.float32),
            restrained_grid,
            [],
            params,
        ),
        method=params.refine_method,
        options={"maxiter": 18, "xtol": 0.2, "ftol": 0.2}
        if params.refine_method == "Powell"
        else {"maxiter": 32, "xatol": 0.2, "fatol": 0.2},
    )
    refined_score = _score_segment_pose(
        chain,
        seed_points,
        seed_weights,
        np.asarray(result.x, dtype=np.float32),
        context,
        params,
        restrained_grid,
    )
    if segment_stats is not None and _should_revert_segment_refine(refined_score, segment_stats, params):
        return x0, float(score00)
    return np.asarray(result.x, dtype=np.float32), float(refined_score)


def _bundle_from_segment_solutions(
    chain: Chain,
    pose: Pose,
    pose_rank: int,
    seed_segment: int,
    growth_order: tuple[int, ...],
    segment_solutions: np.ndarray,
    context: SearchContext,
    params: Params,
) -> RefinedPoseBundle:
    coords = _compose_segment_coords(chain, segment_solutions)
    segment_scores, total_score, normalized_score, grid_score = _score_refined_bundle(
        chain,
        segment_solutions,
        context,
        params,
    )
    return RefinedPoseBundle(
        pose=pose,
        pose_rank=pose_rank,
        seed_segment=seed_segment,
        growth_order=growth_order,
        segment_solutions=segment_solutions.astype(np.float32, copy=False),
        coords=coords,
        score=float(total_score),
        grid_score=float(grid_score),
        segment_scores=segment_scores.astype(np.float32, copy=False),
        normalized_score=normalized_score,
        branch_family=growth_branch_family(growth_order),
    )


def _select_diverse_seed_candidates(
    candidates: list[SeedScreenCandidate],
    limit: int,
    total_segments: int | None = None,
    params: Params | None = None,
) -> list[SeedScreenCandidate]:
    if limit <= 0 or not candidates:
        return []

    def _seed_rank_key(item: SeedScreenCandidate) -> tuple[float, float, float, int, int]:
        if total_segments is not None and params is not None:
            stage_excess = _progress_stage_excess(item.normalized_score, total_segments, 1, params)
            return (
                stage_excess,
                float("inf") if item.normalized_score is None else item.normalized_score,
                item.score,
                item.pose_rank,
                item.seed_segment,
            )
        return (
            0.0,
            float("inf") if item.normalized_score is None else item.normalized_score,
            item.score,
            item.pose_rank,
            item.seed_segment,
        )

    by_segment: dict[int, list[SeedScreenCandidate]] = {}
    for candidate in sorted(candidates, key=_seed_rank_key):
        by_segment.setdefault(candidate.seed_segment, []).append(candidate)

    selected: list[SeedScreenCandidate] = []
    selected_keys: set[tuple[int, int]] = set()
    segment_order = sorted(by_segment)
    made_progress = True
    while len(selected) < limit and made_progress:
        made_progress = False
        for seed_segment in segment_order:
            bucket = by_segment[seed_segment]
            while bucket:
                candidate = bucket.pop(0)
                key = (candidate.pose_rank, candidate.seed_segment)
                if key in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add(key)
                made_progress = True
                break
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        for candidate in sorted(candidates, key=_seed_rank_key):
            key = (candidate.pose_rank, candidate.seed_segment)
            if key in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            if len(selected) >= limit:
                break

    return selected


def _bundle_rmsd(bundle_a: RefinedPoseBundle, bundle_b: RefinedPoseBundle) -> float:
    return float(np.sqrt(np.mean(np.sum((bundle_a.coords - bundle_b.coords) ** 2, axis=1))))


def _min_bundle_rmsd(bundle: RefinedPoseBundle, selected: list[RefinedPoseBundle]) -> float:
    if not selected:
        return float("inf")
    return min(_bundle_rmsd(bundle, existing) for existing in selected)


def _bundle_progress_rank_penalty(
    bundle: RefinedPoseBundle,
    total_segments: int | None = None,
    params: Params | None = None,
) -> float:
    penalty = float(bundle.progress_feedback_penalty)
    if total_segments is None or params is None or bundle.progress_stage_count <= 0:
        return penalty

    stage_score = bundle.progress_weighted_normalized_score
    if stage_score is None:
        stage_score = bundle.progress_worst_normalized_score
    if stage_score is None:
        return penalty

    grown_segments = min(max(bundle.progress_stage_count, 1), total_segments)
    penalty += _progress_stage_excess(stage_score, total_segments, grown_segments, params)
    if bundle.progress_stage_scores:
        best_stage = float(bundle.progress_stage_scores[0])
        for stage_idx, history_score in enumerate(bundle.progress_stage_scores, start=1):
            stage_penalty = 0.25 * _progress_stage_excess(history_score, total_segments, stage_idx, params)
            if stage_idx > 1 and history_score > best_stage:
                stage_penalty += 0.10 * (history_score - best_stage) * (stage_idx / max(total_segments, 1))
            penalty += stage_penalty
            best_stage = min(best_stage, float(history_score))
        penalty += _tail_stage_regression_penalty(bundle, total_segments, params)
        penalty += _near_cutoff_stage_penalty(bundle, total_segments, params)
        penalty += _terminal_full_regression_penalty(bundle, total_segments, params)
        penalty += _terminal_full_history_drag_penalty(bundle, total_segments)
    penalty += _incomplete_growth_family_penalty(bundle, total_segments)
    if total_segments > 1 and len(bundle.growth_order) > bundle.progress_stage_count:
        pending = len(bundle.growth_order) - bundle.progress_stage_count
        penalty += 0.05 * (pending / max(total_segments - 1, 1))
    return float(penalty)


def _incomplete_growth_family_penalty(bundle: RefinedPoseBundle, total_segments: int) -> float:
    grown_segments = len(bundle.growth_order)
    if total_segments <= 1 or grown_segments <= 0 or grown_segments >= total_segments:
        return 0.0
    remaining_fraction = (total_segments - grown_segments) / max(total_segments - 1, 1)
    if grown_segments == 1:
        return float(0.03 * remaining_fraction)
    progressed_fraction = (grown_segments - 1) / max(total_segments - 1, 1)
    return float(0.04 + 0.04 * remaining_fraction + 0.02 * progressed_fraction)


def _tail_stage_regression_penalty(
    bundle: RefinedPoseBundle,
    total_segments: int,
    params: Params,
) -> float:
    if len(bundle.progress_stage_scores) < 2:
        return 0.0
    tail_score = float(bundle.progress_stage_scores[-1])
    prev_score = float(bundle.progress_stage_scores[-2])
    if tail_score <= prev_score:
        return 0.0
    tail_excess = _progress_stage_excess(tail_score, total_segments, len(bundle.progress_stage_scores), params)
    if tail_excess <= 0.0:
        return 0.0
    regression = tail_score - prev_score
    return float(0.08 * tail_excess + 0.04 * regression)


def _near_cutoff_stage_penalty(
    bundle: RefinedPoseBundle,
    total_segments: int,
    params: Params,
) -> float:
    if len(bundle.progress_stage_scores) < 2:
        return 0.0
    penalty = 0.0
    tail_count = min(2, len(bundle.progress_stage_scores))
    for offset, history_score in enumerate(bundle.progress_stage_scores[-tail_count:], start=1):
        stage_idx = len(bundle.progress_stage_scores) - tail_count + offset
        cutoff = _progress_stage_cutoff(total_segments, stage_idx, params)
        margin = cutoff - float(history_score)
        if margin <= 0.0 or margin >= 0.25:
            continue
        stage_weight = stage_idx / max(total_segments, 1)
        penalty += 0.03 * ((0.25 - margin) / 0.25) * stage_weight
    return float(penalty)


def _terminal_full_regression_penalty(
    bundle: RefinedPoseBundle,
    total_segments: int,
    params: Params,
) -> float:
    regression = _terminal_full_regression_amount(bundle, total_segments)
    if regression <= 0.0:
        return 0.0
    penalty = 0.16 * regression
    final_score = float(bundle.normalized_score)
    final_margin = _progress_stage_cutoff(total_segments, total_segments, params) - final_score
    if final_margin < 0.25:
        penalty += 0.03 * ((0.25 - max(final_margin, 0.0)) / 0.25)
    return float(penalty)


def _terminal_full_history_drag_amount(bundle: RefinedPoseBundle, total_segments: int | None) -> float:
    if total_segments is None or total_segments <= 1:
        return 0.0
    if len(bundle.growth_order) < total_segments:
        return 0.0
    if bundle.normalized_score is None or bundle.progress_weighted_normalized_score is None:
        return 0.0
    return float(max(float(bundle.normalized_score) - float(bundle.progress_weighted_normalized_score), 0.0))


def _terminal_full_history_drag_penalty(bundle: RefinedPoseBundle, total_segments: int) -> float:
    drag = _terminal_full_history_drag_amount(bundle, total_segments)
    if drag <= 0.0:
        return 0.0
    return float(0.18 * drag)


def _apply_progress_feedback_penalty(
    bundle: RefinedPoseBundle,
    progress_feedback_bias: ProgressFeedbackBias,
) -> RefinedPoseBundle:
    if isinstance(progress_feedback_bias, dict):
        growth_length = len(bundle.growth_order)
        penalty = progress_feedback_bias.get(
            progress_feedback_bucket(
                bundle.growth_order,
                bundle.seed_segment,
                branch_family=bundle.branch_family,
            ),
            0.0,
        )
        if penalty <= 0.0:
            penalty = progress_feedback_bias.get((growth_length, bundle.seed_segment), 0.0)
        if penalty <= 0.0:
            penalty = progress_feedback_bias.get(growth_length, 0.0)
    else:
        penalty = progress_feedback_bias
    bundle.progress_feedback_penalty = float(max(penalty, 0.0))
    return bundle


def _bundle_rank_score(
    bundle: RefinedPoseBundle,
    total_segments: int | None = None,
    params: Params | None = None,
) -> float:
    if bundle.normalized_score is not None:
        base_score = float(bundle.normalized_score)
    else:
        base_score = float(bundle.score)
    return float(base_score + _bundle_progress_rank_penalty(bundle, total_segments=total_segments, params=params))


def _bundle_family_state(bundle: RefinedPoseBundle, total_segments: int | None) -> str:
    if total_segments is None or total_segments <= 1:
        return "generic"
    grown_segments = len(bundle.growth_order)
    if grown_segments >= total_segments:
        return "full"
    if grown_segments <= 1:
        return "seed"
    return "snapshot"


def _bundle_family_preference_margin(preferred_state: str, other_state: str) -> float:
    if preferred_state == other_state:
        return 0.0
    if preferred_state == "full":
        if other_state == "snapshot":
            return 0.12
        if other_state == "seed":
            return 0.08
    if preferred_state == "seed" and other_state == "snapshot":
        return 0.04
    return 0.0


def _terminal_full_regression_amount(bundle: RefinedPoseBundle, total_segments: int | None) -> float:
    if total_segments is None or total_segments <= 1:
        return 0.0
    if bundle.normalized_score is None or not bundle.progress_stage_scores:
        return 0.0
    if len(bundle.growth_order) < total_segments:
        return 0.0
    last_stage_score = float(bundle.progress_stage_scores[-1])
    final_score = float(bundle.normalized_score)
    return float(max(final_score - last_stage_score, 0.0))


def _prefer_family_representative(
    preferred: RefinedPoseBundle,
    other: RefinedPoseBundle,
    preferred_rank: float,
    other_rank: float,
    total_segments: int | None,
) -> bool:
    preferred_state = _bundle_family_state(preferred, total_segments)
    other_state = _bundle_family_state(other, total_segments)
    margin = _bundle_family_preference_margin(preferred_state, other_state)
    if preferred_state == "full":
        margin = max(margin - 0.5 * _terminal_full_regression_amount(preferred, total_segments), 0.0)
        margin = max(margin - 0.6 * _terminal_full_history_drag_amount(preferred, total_segments), 0.0)
    if margin <= 0.0:
        return False
    return preferred_rank <= (other_rank + margin)


def _bundle_family_sort_adjustment(bundle: RefinedPoseBundle, total_segments: int | None) -> float:
    state = _bundle_family_state(bundle, total_segments)
    if state == "full":
        return float(
            -0.03
            + 0.25 * _terminal_full_regression_amount(bundle, total_segments)
            + 0.35 * _terminal_full_history_drag_amount(bundle, total_segments)
        )
    if state == "seed":
        return -0.01
    return 0.0


def _bundle_selection_rank(
    bundle: RefinedPoseBundle,
    total_segments: int | None = None,
    params: Params | None = None,
) -> float:
    return float(
        _bundle_rank_score(bundle, total_segments=total_segments, params=params)
        + _bundle_family_sort_adjustment(bundle, total_segments)
    )


def _is_near_equivalent_replacement(existing_rank: float, bundle_rank: float, margin: float = 0.12) -> bool:
    improvement = existing_rank - bundle_rank
    return bool(improvement >= 0.0 and improvement <= margin)


def _should_replace_regressed_full_with_snapshot(
    existing: RefinedPoseBundle,
    bundle: RefinedPoseBundle,
    total_segments: int | None,
) -> bool:
    if _bundle_family_state(existing, total_segments) != "full":
        return False
    if _bundle_family_state(bundle, total_segments) != "snapshot":
        return False
    return bool(
        _terminal_full_regression_amount(existing, total_segments) > 0.0
        or _terminal_full_history_drag_amount(existing, total_segments) >= 0.15
    )


def _set_bundle_progress_summary(
    bundle: RefinedPoseBundle,
    progress_scores: list[float],
    pair_progress_scores: list[float | None] | None = None,
) -> RefinedPoseBundle:
    combined_scores = list(progress_scores)
    if pair_progress_scores:
        combined_scores = [
            max(score, pair_score) if pair_score is not None else score
            for score, pair_score in zip(progress_scores, pair_progress_scores, strict=False)
        ]

    bundle.progress_stage_count = len(combined_scores)
    bundle.progress_stage_scores = tuple(float(score) for score in combined_scores)
    bundle.progress_worst_normalized_score = max(combined_scores) if combined_scores else None
    if combined_scores:
        weights = np.arange(1, len(combined_scores) + 1, dtype=np.float32)
        scores = np.asarray(combined_scores, dtype=np.float32)
        bundle.progress_weighted_normalized_score = float(np.dot(scores, weights) / np.sum(weights, dtype=np.float32))
    else:
        bundle.progress_weighted_normalized_score = None
    return bundle


def _coupled_progress_stage_score(
    bundle_normalized_score: float | None,
    child_normalized_score: float | None = None,
    pair_normalized_score: float | None = None,
) -> float | None:
    scores = [
        score
        for score in (bundle_normalized_score, child_normalized_score, pair_normalized_score)
        if score is not None
    ]
    if not scores:
        return None
    return float(max(scores))


def _progress_stage_cutoff(
    total_segments: int,
    grown_segments: int,
    params: Params,
) -> float:
    if total_segments <= 1:
        return float(params.flexible_cutoff_score)
    remaining = max(total_segments - grown_segments, 0)
    relax = params.flexible_progress_score_relax * (remaining / max(total_segments - 1, 1))
    return float(params.flexible_cutoff_score + relax)


def _progress_stage_excess(
    normalized_score: float | None,
    total_segments: int,
    grown_segments: int,
    params: Params,
) -> float:
    if normalized_score is None:
        return 0.0
    return float(max(normalized_score - _progress_stage_cutoff(total_segments, grown_segments, params), 0.0))


def _progress_feedback_stage_load(bundle: RefinedPoseBundle, total_segments: int) -> float:
    if total_segments <= 1:
        return 0.0
    penalty = max(float(bundle.progress_feedback_penalty), 0.0)
    if penalty <= 0.0:
        return 0.0
    grown_segments = min(max(len(bundle.growth_order), 1), total_segments)
    progress_fraction = grown_segments / max(total_segments, 1)
    return float(penalty * (0.15 + 0.35 * progress_fraction))


def _retain_stage_progress_bundles(
    progress_bundles: list[RefinedPoseBundle],
    chain: Chain,
    params: Params,
    stage_scores: list[float | None] | None = None,
) -> tuple[list[RefinedPoseBundle], bool]:
    if not progress_bundles:
        return [], True

    retained: list[RefinedPoseBundle] = []
    ordered = sorted(
        enumerate(progress_bundles),
        key=lambda item: (len(item[1].growth_order), item[1].pose_rank, item[1].seed_segment),
    )
    for idx, bundle in ordered:
        stage_score = bundle.normalized_score
        if stage_scores is not None and idx < len(stage_scores):
            stage_score = stage_scores[idx]
        if stage_score is None:
            retained.append(bundle)
            continue
        stage_load = _progress_stage_excess(stage_score, chain.n_segments, len(bundle.growth_order), params)
        stage_load += _progress_feedback_stage_load(bundle, chain.n_segments)
        if stage_load > 0.0:
            return retained, False
        retained.append(bundle)
    return retained, True


def _prune_redundant_progress_bundles(
    bundles: list[RefinedPoseBundle],
    rmsd_cutoff: float,
    total_segments: int | None = None,
    params: Params | None = None,
) -> list[RefinedPoseBundle]:
    if len(bundles) <= 1:
        return bundles

    kept: list[RefinedPoseBundle] = []
    ordered = sorted(
        bundles,
        key=lambda item: (
            item.pose_rank,
            item.seed_segment,
            -len(item.growth_order),
            _bundle_rank_score(item, total_segments=total_segments, params=params),
            item.score,
        ),
    )
    for bundle in ordered:
        dominated = False
        replacement_idx: int | None = None
        for idx, existing in enumerate(kept):
            if existing.pose_rank != bundle.pose_rank or existing.seed_segment != bundle.seed_segment:
                continue
            if len(existing.growth_order) < len(bundle.growth_order):
                continue
            if _bundle_rmsd(existing, bundle) >= rmsd_cutoff:
                continue
            existing_rank = _bundle_rank_score(existing, total_segments=total_segments, params=params)
            bundle_rank = _bundle_rank_score(bundle, total_segments=total_segments, params=params)
            if _should_replace_regressed_full_with_snapshot(existing, bundle, total_segments):
                replacement_idx = idx
                break
            if _prefer_family_representative(
                existing,
                bundle,
                preferred_rank=existing_rank,
                other_rank=bundle_rank,
                total_segments=total_segments,
            ):
                dominated = True
                break
            existing_selection_rank = _bundle_selection_rank(
                existing,
                total_segments=total_segments,
                params=params,
            )
            bundle_selection_rank = _bundle_selection_rank(
                bundle,
                total_segments=total_segments,
                params=params,
            )
            if existing_rank > bundle_rank:
                if (
                    bundle_selection_rank < existing_selection_rank
                    and _is_near_equivalent_replacement(existing_selection_rank, bundle_selection_rank)
                ):
                    replacement_idx = idx
                    break
                if (
                    bundle_selection_rank == existing_selection_rank
                    and bundle.score < existing.score
                ):
                    replacement_idx = idx
                    break
                continue
            if (
                bundle_selection_rank < existing_selection_rank
                and _is_near_equivalent_replacement(existing_selection_rank, bundle_selection_rank)
            ):
                replacement_idx = idx
                break
            if bundle_selection_rank == existing_selection_rank and bundle.score < existing.score:
                replacement_idx = idx
                break
            if existing_rank == bundle_rank and existing.score > bundle.score:
                continue
            dominated = True
            break
        if not dominated:
            if replacement_idx is not None:
                kept[replacement_idx] = bundle
                continue
            kept.append(bundle)

    return kept


def _select_diverse_bundles(
    bundles: list[RefinedPoseBundle],
    limit: int,
    rmsd_cutoff: float,
    score_cutoff: float | None = None,
    min_keep: int = 1,
    total_segments: int | None = None,
    params: Params | None = None,
) -> list[RefinedPoseBundle]:
    if limit <= 0 or not bundles:
        return []

    selected: list[RefinedPoseBundle] = []
    min_keep = max(1, min(limit, min_keep))
    ordered = sorted(
        bundles,
        key=lambda item: (
            _bundle_selection_rank(item, total_segments=total_segments, params=params),
            item.score,
            -len(item.growth_order),
            item.pose_rank,
            item.seed_segment,
        ),
    )
    for bundle in ordered:
        if (
            score_cutoff is not None
            and len(selected) >= min_keep
            and _bundle_rank_score(bundle, total_segments=total_segments, params=params) > score_cutoff
        ):
            break
        replacement_idx: int | None = None
        bundle_rank = _bundle_rank_score(bundle, total_segments=total_segments, params=params)
        bundle_selection_rank = _bundle_selection_rank(bundle, total_segments=total_segments, params=params)
        for idx, existing in enumerate(selected):
            if _bundle_rmsd(existing, bundle) >= rmsd_cutoff:
                continue
            existing_rank = _bundle_rank_score(existing, total_segments=total_segments, params=params)
            existing_selection_rank = _bundle_selection_rank(
                existing,
                total_segments=total_segments,
                params=params,
            )
            if _prefer_family_representative(
                bundle,
                existing,
                preferred_rank=bundle_rank,
                other_rank=existing_rank,
                total_segments=total_segments,
            ):
                replacement_idx = idx
                break
            if bundle_selection_rank < existing_selection_rank:
                replacement_idx = idx
                break
            if bundle_selection_rank == existing_selection_rank:
                if bundle.score < existing.score:
                    replacement_idx = idx
                    break
                if bundle.score == existing.score and len(bundle.growth_order) > len(existing.growth_order):
                    replacement_idx = idx
                    break
            replacement_idx = -1
            break
        if replacement_idx == -1:
            continue
        if replacement_idx is not None:
            selected[replacement_idx] = bundle
            continue
        selected.append(bundle)
        if len(selected) >= limit:
            break

    return selected


def _growth_occupancy_penalty(points_world: np.ndarray, grown_coords: np.ndarray, params: Params) -> float:
    if points_world.size == 0 or grown_coords.size == 0:
        return 0.0
    rsigma2 = (math.pi / (2.4 + 0.8 * params.resol)) ** 2
    bw = math.sqrt(6.0 / rsigma2)
    tree = cKDTree(grown_coords)
    dists, _ = tree.query(points_world, distance_upper_bound=bw)
    valid = np.isfinite(dists)
    if not np.any(valid):
        return 0.0
    scores = np.zeros(len(points_world), dtype=np.float32)
    d2s = dists[valid] ** 2
    scores[valid] = np.exp(-rsigma2 * d2s).astype(np.float32, copy=False)
    return float(np.mean(scores))


def _growth_suppressed_refinement_grid(
    context: SearchContext,
    grown_coords: np.ndarray,
    params: Params,
) -> np.ndarray:
    if grown_coords.size == 0:
        return context.refinement_grid
    centered = (grown_coords - context.centrioda).astype(np.float32, copy=False)
    weights = np.ones(len(centered), dtype=np.float32)
    suppression = _build_smoothed_grid(
        centered,
        weights,
        context.slowera,
        params.sgrid,
        context.nxyz0,
        params.resol,
    )
    return (context.refinement_grid + suppression).astype(np.float32, copy=False)


def _growth_stage_intensity(
    total_segments: int,
    current_segments: set[int],
    active_segments: set[int],
    params: Params,
    rigid_score: float | None = None,
    rigid_score_mean: float | None = None,
    rigid_score_std: float | None = None,
) -> float:
    covered = len(current_segments | active_segments)
    remaining = max(total_segments - covered, 0)
    intensity = params.growth_restraint_power + params.growth_restraint_step_scale * remaining
    if rigid_score is not None and rigid_score_mean is not None and rigid_score_std is not None and rigid_score_std > 1e-6:
        zscore = (rigid_score - rigid_score_mean) / rigid_score_std
        scale = 1.0 / max(abs(zscore), params.growth_pose_z_min_abs)
        scale = float(np.clip(scale, params.growth_pose_intensity_floor, params.growth_pose_intensity_cap))
        intensity *= scale
    return float(max(intensity, 1e-3))


def _restrained_refinement_grid(
    context: SearchContext,
    points_world: np.ndarray,
    base_grid: np.ndarray,
    params: Params,
    intensity: float,
) -> np.ndarray:
    if points_world.size == 0 or intensity <= 0.0:
        return base_grid

    grid_points = ((points_world - context.centrioda - context.slowera) / params.sgrid).astype(np.float32, copy=False)
    rsigma2 = (math.pi * params.sgrid / (2.4 + 0.8 * params.resol)) ** 2
    bw = math.sqrt(6.0 / rsigma2) + params.clash_dist / params.sgrid
    radius = bw * 2.0
    lower = np.floor(np.min(grid_points, axis=0) - radius).astype(np.int32)
    upper = np.ceil(np.max(grid_points, axis=0) + radius).astype(np.int32)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, context.nxyz0 - 1)
    if np.any(upper < lower):
        return base_grid

    block_shape = tuple((upper - lower + 1).tolist())
    occupancy = np.ones(block_shape, dtype=np.uint8)
    local_points = np.rint(grid_points - lower.astype(np.float32)).astype(np.int32)
    valid_points = np.all((local_points >= 0) & (local_points < np.asarray(block_shape, dtype=np.int32)), axis=1)
    local_points = local_points[valid_points]
    if len(local_points) == 0:
        return base_grid
    occupancy[local_points[:, 0], local_points[:, 1], local_points[:, 2]] = 0

    dists = ndi.distance_transform_edt(occupancy).astype(np.float32, copy=False)
    valid = dists <= radius
    if not np.any(valid):
        return base_grid

    cdist = params.clash_dist / params.sgrid
    d2s = np.maximum(dists[valid] - cdist, 0.0) ** 2
    probs = np.exp(-(rsigma2 * d2s / max(params.growth_restraint_rscale, 1e-6)) * intensity).astype(
        np.float32, copy=False
    )
    mask = np.zeros_like(base_grid, dtype=np.float32)
    mask_block = mask[lower[0] : upper[0] + 1, lower[1] : upper[1] + 1, lower[2] : upper[2] + 1]
    mask_block[valid] = probs
    return (base_grid * mask).astype(np.float32, copy=False)


def _segment_suppression_grid(
    context: SearchContext,
    segment_coords: np.ndarray,
    params: Params,
) -> np.ndarray:
    if segment_coords.size == 0:
        return np.zeros_like(context.refinement_grid)
    centered = (segment_coords - context.centrioda).astype(np.float32, copy=False)
    weights = np.ones(len(centered), dtype=np.float32)
    return _build_smoothed_grid(
        centered,
        weights,
        context.slowera,
        params.sgrid,
        context.nxyz0,
        params.resol,
    )


def _active_growth_state(
    assembled_coords: np.ndarray,
    grown_segment_coords: dict[int, np.ndarray],
    suppression_grids: dict[int, np.ndarray],
    excluded_segments: set[int],
    context: SearchContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_blocks: list[np.ndarray] = []
    if assembled_coords.size:
        fixed_blocks.append(assembled_coords)

    grown_blocks: list[np.ndarray] = []
    local_grid = context.refinement_grid.copy()
    for segment, coords in grown_segment_coords.items():
        if segment in excluded_segments:
            continue
        fixed_blocks.append(coords)
        grown_blocks.append(coords)
        local_grid += suppression_grids[segment]

    fixed_coords = np.concatenate(fixed_blocks, axis=0) if fixed_blocks else np.empty((0, 3), dtype=np.float32)
    grown_coords = np.concatenate(grown_blocks, axis=0) if grown_blocks else np.empty((0, 3), dtype=np.float32)
    return fixed_coords, grown_coords, local_grid


def _combined_restraints(restraints: dict[int, list[SegmentRestraint]], segments: tuple[int, ...]) -> list[SegmentRestraint]:
    combined: list[SegmentRestraint] = []
    for segment in segments:
        combined.extend(restraints.get(segment, []))
    return combined


def _filter_restraints_by_active_segments(
    restraints: list[SegmentRestraint],
    active_segments: set[int] | None,
) -> list[SegmentRestraint]:
    if active_segments is None:
        return restraints
    return [restraint for restraint in restraints if restraint.neighbor_segment in active_segments]


def _pair_refine_seed(
    chain: Chain,
    parent_segment: int,
    child_segment: int,
    solutions: np.ndarray,
    base_solution: np.ndarray,
    segment_indices: dict[int, np.ndarray],
    context: SearchContext,
    assembled_coords: np.ndarray,
    restraints: dict[int, list[SegmentRestraint]],
    active_segments: set[int],
    grown_segment_coords: dict[int, np.ndarray],
    suppression_grids: dict[int, np.ndarray],
    segment_stats: dict[int, SegmentScoreStats],
    base_pose_score: float | None,
    params: Params,
) -> tuple[np.ndarray, float]:
    pair_ids = np.concatenate((segment_indices[parent_segment], segment_indices[child_segment]))
    pair_points = chain.coords[pair_ids]
    pair_weights = chain.weights[pair_ids]

    fixed_coords, grown_coords, local_grid = _active_growth_state(
        assembled_coords,
        grown_segment_coords,
        suppression_grids,
        {parent_segment, child_segment},
        context,
    )

    x0 = solutions[parent_segment - 1].copy()
    pair_world = transform_points(pair_points, chain.centroid, x0)
    restrained_grid = _restrained_refinement_grid(
        context,
        pair_world,
        local_grid,
        params,
        intensity=_growth_stage_intensity(
            chain.n_segments,
            {parent_segment, child_segment},
            active_segments,
            params,
            rigid_score=base_pose_score,
            rigid_score_mean=chain.rigid_score_mean,
            rigid_score_std=chain.rigid_score_std,
        ),
    )
    score00 = _score_segment_pose(chain, pair_points, pair_weights, x0, context, params, restrained_grid)
    pair_restraints = _filter_restraints_by_active_segments(
        _combined_restraints(restraints, (parent_segment, child_segment)),
        active_segments | {parent_segment, child_segment},
    )
    result = optimize.minimize(
        _segment_objective,
        x0=x0,
        args=(
            base_solution.copy(),
            pair_points,
            pair_weights,
            chain.centroid,
            context,
            fixed_coords,
            grown_coords,
            restrained_grid,
            pair_restraints,
            params,
        ),
        method=params.refine_method,
        options={"maxiter": 30, "xtol": 0.1, "ftol": 0.1}
        if params.refine_method == "Powell"
        else {"maxiter": 60, "xatol": 0.1, "fatol": 0.1},
    )
    refined = np.asarray(result.x, dtype=np.float32)
    refined_score = _score_segment_pose(chain, pair_points, pair_weights, refined, context, params, restrained_grid)
    if _should_revert_pair_refine(refined_score, segment_stats[child_segment], segment_stats[parent_segment], params):
        return (
            x0,
            _pair_refine_normalized_score(score00, segment_stats[child_segment], segment_stats[parent_segment]),
        )
    if refined_score > score00:
        return (
            x0,
            _pair_refine_normalized_score(score00, segment_stats[child_segment], segment_stats[parent_segment]),
        )
    return (
        refined,
        _pair_refine_normalized_score(refined_score, segment_stats[child_segment], segment_stats[parent_segment]),
    )


def refine_chain_segments(
    chain: Chain,
    base_solution: np.ndarray,
    context: SearchContext,
    assembled_coords: np.ndarray,
    params: Params,
    pass_orders: list[list[int]] | None = None,
    growth_parents: dict[int, int] | None = None,
    segment_stats: dict[int, SegmentScoreStats] | None = None,
    base_pose_score: float | None = None,
    progress_snapshots: list[GrowthStageSnapshot | tuple[tuple[int, ...], np.ndarray]] | None = None,
    initial_solutions: np.ndarray | None = None,
    pre_grown_segments: set[int] | None = None,
) -> np.ndarray:
    solutions = np.tile(base_solution.astype(np.float32, copy=False), (chain.n_segments, 1))
    if initial_solutions is not None:
        copied = np.asarray(initial_solutions, dtype=np.float32)
        if copied.shape == solutions.shape:
            solutions = copied.copy()
    if chain.n_segments <= 1:
        return solutions

    segment_indices = _segment_index_map(chain)
    if segment_stats is None:
        segment_stats = {
            segment: SegmentScoreStats(mean=0.0, std=max(params.flexible_revert_min_std, 1e-3))
            for segment in range(1, chain.n_segments + 1)
        }
    if pass_orders is not None:
        refine_orders = pass_orders
    else:
        base_orders = _segment_orders(chain)
        refine_orders = [base_orders[idx % len(base_orders)] for idx in range(params.flexible_refine_passes)]
    grown_segments: set[int] = set() if pre_grown_segments is None else {int(segment) for segment in pre_grown_segments}
    grown_segment_coords: dict[int, np.ndarray] = {}
    suppression_grids: dict[int, np.ndarray] = {}
    growth_progress: list[int] = [] if pre_grown_segments is None else sorted(int(segment) for segment in pre_grown_segments)
    if growth_parents is not None:
        for segment in growth_progress:
            grown_segment_coords[segment] = _segment_world_coords(chain, segment_indices, segment, solutions)
            suppression_grids[segment] = _segment_suppression_grid(context, grown_segment_coords[segment], params)

    with stage_timer(f"chain {chain.index:02d} flexible segment refinement"):
        for pass_idx, order in enumerate(refine_orders):
            log_message(
                f"chain {chain.index:02d} flexible pass {pass_idx + 1}/{len(refine_orders)}: "
                f"order={order}"
            )
            for segment in order:
                if growth_parents is not None and segment in grown_segments and segment not in growth_parents:
                    continue
                ids = segment_indices[segment]
                segment_points = chain.coords[ids]
                segment_weights = chain.weights[ids]
                all_restraints = _build_segment_restraints(chain, solutions, base_solution, segment_indices)
                active_segments = grown_segments if growth_parents is not None else None
                restraints = _filter_restraints_by_active_segments(all_restraints[segment], active_segments)

                if growth_parents is not None:
                    fixed_coords, grown_coords, local_grid = _active_growth_state(
                        assembled_coords,
                        grown_segment_coords,
                        suppression_grids,
                        {segment},
                        context,
                    )
                else:
                    fixed_blocks: list[np.ndarray] = []
                    if assembled_coords.size:
                        fixed_blocks.append(assembled_coords)
                    for other in range(1, chain.n_segments + 1):
                        if other == segment:
                            continue
                        other_ids = segment_indices[other]
                        fixed_blocks.append(transform_points(chain.coords[other_ids], chain.centroid, solutions[other - 1]))
                    fixed_coords = np.concatenate(fixed_blocks, axis=0) if fixed_blocks else np.empty((0, 3), dtype=np.float32)
                    grown_coords = np.empty((0, 3), dtype=np.float32)
                    local_grid = context.refinement_grid

                pair_progress_score: float | None = None
                if growth_parents is not None and segment in growth_parents and growth_parents[segment] in grown_segments:
                    x0, pair_progress_score = _pair_refine_seed(
                        chain,
                        parent_segment=growth_parents[segment],
                        child_segment=segment,
                        solutions=solutions,
                        base_solution=base_solution,
                        segment_indices=segment_indices,
                        context=context,
                        assembled_coords=assembled_coords,
                        restraints=all_restraints,
                        active_segments=grown_segments,
                        grown_segment_coords=grown_segment_coords,
                        suppression_grids=suppression_grids,
                        segment_stats=segment_stats,
                        base_pose_score=base_pose_score,
                        params=params,
                    )
                else:
                    x0 = solutions[segment - 1].copy()
                segment_world = transform_points(segment_points, chain.centroid, x0)
                restrained_grid = _restrained_refinement_grid(
                    context,
                    segment_world,
                    local_grid,
                    params,
                    intensity=_growth_stage_intensity(
                        chain.n_segments,
                        {segment},
                        grown_segments if growth_parents is not None else set(),
                        params,
                        rigid_score=base_pose_score,
                        rigid_score_mean=chain.rigid_score_mean,
                        rigid_score_std=chain.rigid_score_std,
                    ),
                )
                result = optimize.minimize(
                    _segment_objective,
                    x0=x0,
                    args=(
                        x0.copy(),
                        segment_points,
                        segment_weights,
                        chain.centroid,
                        context,
                        fixed_coords,
                        grown_coords,
                        restrained_grid,
                        restraints,
                        params,
                    ),
                    method=params.refine_method,
                    options={"maxiter": 40, "xtol": 0.1, "ftol": 0.1}
                    if params.refine_method == "Powell"
                    else {"maxiter": 80, "xatol": 0.1, "fatol": 0.1},
                )
                refined = np.asarray(result.x, dtype=np.float32)
                score00 = _score_segment_pose(chain, segment_points, segment_weights, x0, context, params, restrained_grid)
                refined_score = _score_segment_pose(chain, segment_points, segment_weights, refined, context, params, restrained_grid)
                if _should_revert_segment_refine(refined_score, segment_stats[segment], params) or refined_score > score00:
                    solutions[segment - 1] = x0
                    selected_score = float(score00)
                else:
                    solutions[segment - 1] = refined
                    selected_score = float(refined_score)
                if growth_parents is not None:
                    grown_segment_coords[segment] = _segment_world_coords(chain, segment_indices, segment, solutions)
                    suppression_grids[segment] = _segment_suppression_grid(context, grown_segment_coords[segment], params)
                grown_segments.add(segment)
                if growth_parents is not None and pass_idx == 1 and segment not in growth_progress:
                    growth_progress.append(segment)
                    if progress_snapshots is not None and len(growth_progress) >= 2:
                        progress_snapshots.append(
                            GrowthStageSnapshot(
                                growth_order=tuple(growth_progress),
                                segment_solutions=solutions.astype(np.float32, copy=True),
                                child_normalized_score=_segment_zscore(selected_score, segment_stats[segment]),
                                pair_normalized_score=pair_progress_score,
                            )
                        )

    return solutions


def generate_refined_pose_bundles(
    chain: Chain,
    poses: list[Pose],
    context: SearchContext,
    assembled_coords: np.ndarray,
    params: Params,
    max_bundles: int = 3,
    progress_feedback_bias: ProgressFeedbackBias = 0.0,
) -> list[RefinedPoseBundle]:
    if chain.n_segments <= 1 or not poses or max_bundles <= 0:
        return []

    bundles: list[RefinedPoseBundle] = []
    seed_segments = _seed_segment_candidates(chain, params.flexible_seed_segments)
    n_seed_poses = min(len(poses), params.flexible_seed_poses)
    seed_poses = poses[:n_seed_poses]
    segment_indices = _segment_index_map(chain)
    segment_stats = _estimate_segment_score_stats(chain, seed_poses, segment_indices, context, params)

    for pose_rank, pose in enumerate(seed_poses):
        rigid_segment_solutions = np.tile(pose.solution.astype(np.float32, copy=False), (chain.n_segments, 1))
        rigid_bundle = _bundle_from_segment_solutions(
            chain,
            pose,
            pose_rank=pose_rank,
            seed_segment=0,
            growth_order=(),
            segment_solutions=rigid_segment_solutions,
            context=context,
            params=params,
        )
        bundles.append(_set_bundle_progress_summary(rigid_bundle, []))

    seed_candidates: list[SeedScreenCandidate] = []
    with stage_timer(f"chain {chain.index:02d} flexible seed screening"):
        for pose_rank, pose in enumerate(seed_poses):
            for seed_segment in seed_segments:
                growth_order, growth_parents = _seed_growth_tree(chain, seed_segment)
                seed_solution, score = _refine_seed_segment(
                    chain,
                    pose,
                    seed_segment,
                    context,
                    assembled_coords,
                    params,
                    segment_stats=segment_stats[seed_segment],
                )
                normalized_score = (
                    _segment_zscore(score, segment_stats[seed_segment])
                    if seed_segment in segment_stats
                    else None
                )
                seed_candidates.append(
                    SeedScreenCandidate(
                        pose=pose,
                        pose_rank=pose_rank,
                        seed_segment=seed_segment,
                        growth_order=growth_order,
                        growth_parents=growth_parents,
                        score=score,
                        normalized_score=normalized_score,
                        seed_solution=seed_solution,
                    )
                )
    screen_limit = min(len(seed_candidates), max(max_bundles, 1) * max(params.flexible_screen_factor, 1))
    selected_candidates = _select_diverse_seed_candidates(
        seed_candidates,
        screen_limit,
        total_segments=chain.n_segments,
        params=params,
    )
    log_message(
        f"chain {chain.index:02d}: screened {len(seed_candidates)} seed combination(s), "
        f"refining top {len(selected_candidates)} diverse combination(s)"
    )

    with stage_timer(f"chain {chain.index:02d} flexible candidate generation"):
        for candidate in selected_candidates:
            progress_scores: list[float] = []
            pair_progress_scores: list[float | None] = []
            if candidate.seed_solution is not None:
                seed_segment_solutions = np.tile(
                    candidate.pose.solution.astype(np.float32, copy=False),
                    (chain.n_segments, 1),
                )
                seed_segment_solutions[candidate.seed_segment - 1] = candidate.seed_solution.astype(
                    np.float32,
                    copy=False,
                )
                seed_bundle = _bundle_from_segment_solutions(
                    chain,
                    candidate.pose,
                    pose_rank=candidate.pose_rank,
                    seed_segment=candidate.seed_segment,
                    growth_order=(candidate.seed_segment,),
                    segment_solutions=seed_segment_solutions,
                    context=context,
                    params=params,
                )
                seed_bundle = _apply_progress_feedback_penalty(seed_bundle, progress_feedback_bias)
                if seed_bundle.normalized_score is not None:
                    progress_scores.append(seed_bundle.normalized_score)
                bundles.append(
                    _set_bundle_progress_summary(
                        seed_bundle,
                        progress_scores.copy(),
                        pair_progress_scores.copy(),
                    )
                )

            pass_orders = [[candidate.seed_segment], candidate.growth_order, candidate.growth_order[::-1]]
            progress_snapshots: list[GrowthStageSnapshot | tuple[tuple[int, ...], np.ndarray]] = []
            initial_growth_solutions = np.tile(
                candidate.pose.solution.astype(np.float32, copy=False),
                (chain.n_segments, 1),
            )
            if candidate.seed_solution is not None:
                initial_growth_solutions[candidate.seed_segment - 1] = candidate.seed_solution.astype(
                    np.float32,
                    copy=False,
                )
            segment_solutions = refine_chain_segments(
                chain,
                candidate.pose.solution,
                context,
                assembled_coords,
                params,
                pass_orders=pass_orders,
                growth_parents=candidate.growth_parents,
                segment_stats=segment_stats,
                base_pose_score=candidate.pose.score,
                progress_snapshots=progress_snapshots,
                initial_solutions=initial_growth_solutions,
                pre_grown_segments={candidate.seed_segment},
            )
            progress_bundles: list[RefinedPoseBundle] = []
            stage_scores: list[float | None] = []
            stage_pair_scores: list[float | None] = []
            for snapshot in progress_snapshots:
                if isinstance(snapshot, GrowthStageSnapshot):
                    progress_order = snapshot.growth_order
                    progress_solutions = snapshot.segment_solutions
                    child_normalized_score = snapshot.child_normalized_score
                    pair_normalized_score = snapshot.pair_normalized_score
                else:
                    progress_order, progress_solutions = snapshot
                    child_normalized_score = None
                    pair_normalized_score = None
                progress_bundle = _bundle_from_segment_solutions(
                    chain,
                    candidate.pose,
                    pose_rank=candidate.pose_rank,
                    seed_segment=candidate.seed_segment,
                    growth_order=progress_order,
                    segment_solutions=progress_solutions,
                    context=context,
                    params=params,
                )
                progress_bundle = _apply_progress_feedback_penalty(progress_bundle, progress_feedback_bias)
                progress_bundles.append(progress_bundle)
                stage_scores.append(
                    _coupled_progress_stage_score(
                        progress_bundle.normalized_score,
                        child_normalized_score,
                        pair_normalized_score,
                    )
                )
                stage_pair_scores.append(pair_normalized_score)
            retained_progress_bundles, allow_full_growth = _retain_stage_progress_bundles(
                progress_bundles,
                chain,
                params,
                stage_scores=stage_scores,
            )
            for progress_bundle, stage_score, pair_score in zip(
                retained_progress_bundles,
                stage_scores[: len(retained_progress_bundles)],
                stage_pair_scores[: len(retained_progress_bundles)],
                strict=True,
            ):
                if stage_score is not None:
                    progress_scores.append(stage_score)
                    pair_progress_scores.append(pair_score)
                bundles.append(
                    _set_bundle_progress_summary(
                        progress_bundle,
                        progress_scores.copy(),
                        pair_progress_scores.copy(),
                    )
                )
            if not allow_full_growth:
                log_message(
                    f"chain {chain.index:02d} stopping candidate growth for seed segment {candidate.seed_segment} "
                    f"after stage {len(progress_bundles[len(retained_progress_bundles)].growth_order)} "
                    f"failed stage cutoff"
                )
                continue
            full_bundle = _bundle_from_segment_solutions(
                chain,
                candidate.pose,
                pose_rank=candidate.pose_rank,
                seed_segment=candidate.seed_segment,
                growth_order=tuple(candidate.growth_order),
                segment_solutions=segment_solutions,
                context=context,
                params=params,
            )
            full_bundle = _apply_progress_feedback_penalty(full_bundle, progress_feedback_bias)
            bundles.append(_set_bundle_progress_summary(full_bundle, progress_scores.copy(), pair_progress_scores.copy()))

    pruned_bundles = _prune_redundant_progress_bundles(
        bundles,
        params.rmsdcut2,
        total_segments=chain.n_segments,
        params=params,
    )
    if len(pruned_bundles) != len(bundles):
        log_message(
            f"chain {chain.index:02d}: pruned {len(bundles) - len(pruned_bundles)} redundant intermediate bundle(s)"
        )

    unique_bundles = _select_diverse_bundles(
        pruned_bundles,
        max_bundles,
        params.rmsdcut2,
        score_cutoff=params.flexible_cutoff_score,
        min_keep=params.flexible_nleast,
        total_segments=chain.n_segments,
        params=params,
    )

    log_message(
        f"chain {chain.index:02d}: generated {len(unique_bundles)} flexible candidate bundle(s) "
        f"from {n_seed_poses} rigid seed(s) and {len(seed_segments)} seed segment(s)"
        f" (kept from {len(pruned_bundles)} after intermediate pruning and score cutoff {params.flexible_cutoff_score:.2f}, "
        f"nleast={min(params.flexible_nleast, max_bundles)})"
    )
    return unique_bundles
