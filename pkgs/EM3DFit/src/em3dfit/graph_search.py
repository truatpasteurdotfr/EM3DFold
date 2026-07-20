from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import inf
from typing import TYPE_CHECKING

import numpy as np
from ortools.sat.python import cp_model

from em3dfit.utils import log_message as _base_log_message, stage_timer as _base_stage_timer

if TYPE_CHECKING:
    pass

LOG_STAGE = "Graph"
log_message = partial(_base_log_message, stage=LOG_STAGE)
stage_timer = partial(_base_stage_timer, stage=LOG_STAGE)

GRAPH_OBJECTIVE_SCALE = 100
DEFAULT_RANDOM_SEED = 42


@dataclass(slots=True)
class GraphVertex:
    index: int
    chain_idx: int
    pose_idx: int
    score: float


@dataclass(slots=True)
class CompatibilityGraph:
    vertices: list[GraphVertex]
    adjacency: list[set[int]]
    edge_penalties: list[dict[int, float]]
    chain_count: int
    edge_count: int


@dataclass(slots=True)
class CliqueSearchStats:
    expanded_nodes: int = 0
    cliques_found: int = 0
    pruned_by_bound: int = 0
    pruned_by_chain_cover: int = 0
    prepruned_vertices: int = 0
    best_size: int = 0
    best_score: float = inf
    initial_size: int = 0
    initial_score: float = inf
    initial_vertex_score: float = inf
    initial_edge_penalty: float = 0.0
    best_vertex_score: float = inf
    best_edge_penalty: float = 0.0
    graph_edge_penalty_count: int = 0
    graph_edge_penalty_mean: float = 0.0
    graph_edge_penalty_max: float = 0.0


def _scaled_graph_objective(value: float) -> int:
    return int(round(float(value) * GRAPH_OBJECTIVE_SCALE))


def build_compatibility_graph(
    pose_counts: list[int],
    pose_scores: list[list[float]],
    pair_clash_lookup,
    clash_cutoff: float,
    max_vertices: int,
    pair_penalty_lookup=None,
) -> CompatibilityGraph:
    vertices: list[GraphVertex] = []
    for chain_idx, count in enumerate(pose_counts):
        for pose_idx in range(min(count, max_vertices)):
            vertices.append(
                GraphVertex(
                    index=len(vertices),
                    chain_idx=chain_idx,
                    pose_idx=pose_idx,
                    score=float(pose_scores[chain_idx][pose_idx]),
                )
            )

    adjacency = [set() for _ in vertices]
    edge_penalties = [{} for _ in vertices]
    edge_count = 0
    for i, vi in enumerate(vertices[:-1]):
        for j in range(i + 1, len(vertices)):
            vj = vertices[j]
            if vi.chain_idx == vj.chain_idx:
                continue
            clash = pair_clash_lookup(vi.chain_idx, vi.pose_idx, vj.chain_idx, vj.pose_idx)
            if clash <= clash_cutoff:
                adjacency[i].add(j)
                adjacency[j].add(i)
                penalty = (
                    max(0.0, float(pair_penalty_lookup(vi.chain_idx, vi.pose_idx, vj.chain_idx, vj.pose_idx)))
                    if pair_penalty_lookup is not None
                    else 0.0
                )
                if penalty > 0.0:
                    edge_penalties[i][j] = penalty
                    edge_penalties[j][i] = penalty
                edge_count += 1

    return CompatibilityGraph(
        vertices=vertices,
        adjacency=adjacency,
        edge_penalties=edge_penalties,
        chain_count=len(pose_counts),
        edge_count=edge_count,
    )


def _edge_penalty(graph: CompatibilityGraph, vertex_a: int, vertex_b: int) -> float:
    return float(graph.edge_penalties[vertex_a].get(vertex_b, 0.0))


def _build_cp_sat_model(
    graph: CompatibilityGraph,
    fixed_cost: int | None = None,
    fixed_size: int | None = None,
    maximize_size: bool = False,
    minimize_lex: bool = False,
    hinted_vertices: set[int] | None = None,
) -> tuple[
    cp_model.CpModel,
    list[cp_model.IntVar],
    cp_model.LinearExpr,
    cp_model.LinearExpr,
    cp_model.LinearExpr,
]:
    model = cp_model.CpModel()
    vertex_vars = [model.NewBoolVar(f"v_{idx}") for idx in range(len(graph.vertices))]

    for chain_idx in range(graph.chain_count):
        chain_vertices = [vertex_vars[idx] for idx, vertex in enumerate(graph.vertices) if vertex.chain_idx == chain_idx]
        if chain_vertices:
            model.Add(sum(chain_vertices) <= 1)

    for i, vi in enumerate(graph.vertices[:-1]):
        for j in range(i + 1, len(graph.vertices)):
            vj = graph.vertices[j]
            if vi.chain_idx == vj.chain_idx:
                continue
            if j in graph.adjacency[i]:
                continue
            model.Add(vertex_vars[i] + vertex_vars[j] <= 1)

    objective_terms: list[cp_model.LinearExpr] = []
    for idx, vertex in enumerate(graph.vertices):
        coefficient = _scaled_graph_objective(vertex.score)
        if coefficient != 0:
            objective_terms.append(coefficient * vertex_vars[idx])

    for i, neighbors in enumerate(graph.adjacency):
        for j in neighbors:
            if j <= i:
                continue
            penalty = _edge_penalty(graph, i, j)
            scaled_penalty = _scaled_graph_objective(penalty)
            if scaled_penalty <= 0:
                continue
            edge_var = model.NewBoolVar(f"e_{i}_{j}")
            model.Add(edge_var <= vertex_vars[i])
            model.Add(edge_var <= vertex_vars[j])
            model.Add(edge_var >= vertex_vars[i] + vertex_vars[j] - 1)
            objective_terms.append(scaled_penalty * edge_var)

    objective_expr = sum(objective_terms) if objective_terms else 0
    size_expr = sum(vertex_vars) if vertex_vars else 0
    lex_expr = sum((idx + 1) * vertex_vars[idx] for idx in range(len(vertex_vars))) if vertex_vars else 0

    if fixed_cost is not None:
        model.Add(objective_expr == fixed_cost)
    if fixed_size is not None:
        model.Add(size_expr == fixed_size)
    if maximize_size:
        model.Maximize(size_expr)
    elif minimize_lex:
        model.Minimize(lex_expr)
    else:
        model.Minimize(objective_expr)

    if hinted_vertices is not None:
        hinted = set(hinted_vertices)
        for idx, var in enumerate(vertex_vars):
            model.AddHint(var, 1 if idx in hinted else 0)

    return model, vertex_vars, objective_expr, size_expr, lex_expr


def solve_cp_sat_clique(
    graph: CompatibilityGraph,
    hinted_vertices: set[int] | None = None,
) -> tuple[list[int] | None, int | None, int, int]:
    if not graph.vertices:
        return None, None, 0, 0

    primary_model, primary_vars, objective_expr, _size_expr, _lex_expr = _build_cp_sat_model(
        graph,
        hinted_vertices=hinted_vertices,
    )
    primary_solver = cp_model.CpSolver()
    primary_solver.parameters.num_search_workers = 8
    primary_solver.parameters.random_seed = DEFAULT_RANDOM_SEED
    primary_solver.parameters.log_search_progress = False
    primary_status = primary_solver.Solve(primary_model)
    if primary_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, int(primary_solver.NumBranches()), int(primary_solver.NumConflicts())

    best_cost = int(round(primary_solver.Value(objective_expr)))

    secondary_model, secondary_vars, secondary_cost_expr, secondary_size_expr, _secondary_lex_expr = _build_cp_sat_model(
        graph,
        fixed_cost=best_cost,
        maximize_size=True,
        hinted_vertices=hinted_vertices,
    )
    secondary_solver = cp_model.CpSolver()
    secondary_solver.parameters.num_search_workers = 8
    secondary_solver.parameters.random_seed = DEFAULT_RANDOM_SEED
    secondary_solver.parameters.log_search_progress = False
    secondary_status = secondary_solver.Solve(secondary_model)
    if secondary_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [idx for idx, var in enumerate(primary_vars) if primary_solver.Value(var)]
        return selected, best_cost, int(primary_solver.NumBranches()), int(primary_solver.NumConflicts())

    best_size = int(round(secondary_solver.Value(secondary_size_expr)))

    tertiary_model, tertiary_vars, tertiary_cost_expr, _tertiary_size_expr, _tertiary_lex_expr = _build_cp_sat_model(
        graph,
        fixed_cost=best_cost,
        fixed_size=best_size,
        minimize_lex=True,
        hinted_vertices=hinted_vertices,
    )
    tertiary_solver = cp_model.CpSolver()
    tertiary_solver.parameters.num_search_workers = 8
    tertiary_solver.parameters.random_seed = DEFAULT_RANDOM_SEED
    tertiary_solver.parameters.log_search_progress = False
    tertiary_status = tertiary_solver.Solve(tertiary_model)
    if tertiary_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [idx for idx, var in enumerate(secondary_vars) if secondary_solver.Value(var)]
        branches = int(primary_solver.NumBranches()) + int(secondary_solver.NumBranches())
        conflicts = int(primary_solver.NumConflicts()) + int(secondary_solver.NumConflicts())
        return selected, int(round(secondary_solver.Value(secondary_cost_expr))), branches, conflicts

    selected = [idx for idx, var in enumerate(tertiary_vars) if tertiary_solver.Value(var)]
    branches = (
        int(primary_solver.NumBranches())
        + int(secondary_solver.NumBranches())
        + int(tertiary_solver.NumBranches())
    )
    conflicts = (
        int(primary_solver.NumConflicts())
        + int(secondary_solver.NumConflicts())
        + int(tertiary_solver.NumConflicts())
    )
    return selected, int(round(tertiary_solver.Value(tertiary_cost_expr))), branches, conflicts


def _graph_edge_penalty_stats(graph: CompatibilityGraph) -> tuple[int, float, float]:
    penalties: list[float] = []
    for vertex_idx, neighbors in enumerate(graph.adjacency):
        for neighbor_idx in neighbors:
            if neighbor_idx <= vertex_idx:
                continue
            penalty = _edge_penalty(graph, vertex_idx, neighbor_idx)
            if penalty > 0.0:
                penalties.append(penalty)
    if not penalties:
        return 0, 0.0, 0.0
    return len(penalties), float(np.mean(penalties)), float(np.max(penalties))


def _clique_score_components(graph: CompatibilityGraph, vertices: list[int] | None) -> tuple[float, float, float]:
    if not vertices:
        return 0.0, 0.0, 0.0
    vertex_score = float(sum(graph.vertices[idx].score for idx in vertices))
    edge_penalty = 0.0
    for i, vertex_idx in enumerate(vertices[:-1]):
        for neighbor_idx in vertices[i + 1 :]:
            edge_penalty += _edge_penalty(graph, vertex_idx, neighbor_idx)
    return vertex_score, float(edge_penalty), float(vertex_score + edge_penalty)


def select_best_clique(graph: CompatibilityGraph) -> tuple[list[int] | None, CliqueSearchStats]:
    stats = CliqueSearchStats()
    stats.graph_edge_penalty_count, stats.graph_edge_penalty_mean, stats.graph_edge_penalty_max = _graph_edge_penalty_stats(
        graph
    )

    best_vertices: list[int] | None = None
    with stage_timer("compatibility clique search"):
        best_vertices, _solved_cost, branches, conflicts = solve_cp_sat_clique(graph)
    stats.expanded_nodes = branches
    stats.pruned_by_bound = conflicts
    stats.cliques_found = 1 if best_vertices else 0

    if best_vertices is not None:
        stats.best_size = len(best_vertices)
        stats.best_vertex_score, stats.best_edge_penalty, stats.best_score = _clique_score_components(graph, best_vertices)
        stats.initial_size = stats.best_size
        stats.initial_score = stats.best_score
        stats.initial_vertex_score = stats.best_vertex_score
        stats.initial_edge_penalty = stats.best_edge_penalty

    log_message(
        "graph search stats: "
        f"vertices={len(graph.vertices)}, "
        f"edge_penalties={stats.graph_edge_penalty_count}, "
        f"edge_penalty_mean={stats.graph_edge_penalty_mean:.3f}, "
        f"edge_penalty_max={stats.graph_edge_penalty_max:.3f}, "
        f"initial_size={stats.initial_size}, "
        f"initial_score={stats.initial_score:.3f}, "
        f"best_size={stats.best_size}, "
        f"best_score={stats.best_score:.3f}, "
        f"best_vertex_score={stats.best_vertex_score:.3f}, "
        f"best_edge_penalty={stats.best_edge_penalty:.3f}, "
        f"expanded={stats.expanded_nodes}, "
        f"cliques={stats.cliques_found}, "
        f"prepruned={stats.prepruned_vertices}, "
        f"pruned_bound={stats.pruned_by_bound}, "
        f"pruned_cover={stats.pruned_by_chain_cover}"
    )

    if best_vertices is None:
        return None, stats

    selection = [-1 for _ in range(graph.chain_count)]
    for idx in best_vertices:
        vertex = graph.vertices[idx]
        selection[vertex.chain_idx] = vertex.pose_idx
    return selection, stats


def covered_chain_count(selection: list[int] | None) -> int:
    if selection is None:
        return 0
    return sum(1 for pose_idx in selection if pose_idx >= 0)
