from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Params:
    resol: float
    apix: float = 1.0
    threshold: float = 20.0
    rshift: float = 10.0
    rmerge: float = 1.0
    filter_fraction: float = 0.0
    angle_step: float = 30.0
    fgrid: float = 1.5
    sgrid: float = 0.5
    rmsdcut1: float = 2.5
    rmsdcut2: float = 5.0
    ntop: int = 10
    rigid_nleast: int = 5
    rigid_cutoff_score_early: float = -1.5
    rigid_cutoff_score_late: float = -0.5
    rigid_skip_short_residues: int = 50
    ntrans: int = 8
    clash_dist: float = 0.5
    clash_cutoff: float = 0.05
    external_clash_pair_cutoff: float = 0.15
    external_clash_cutoff: float = 0.30
    external_clash_penalty_weight: float = 2.0
    external_clash_score_cap: float = 0.95
    ldp_coverage_reward_weight: float = 0.20
    graph_external_clash_max_penalty_weight: float = 0.10
    graph_external_clash_max_penalty_cap: float = 0.25
    graph_pair_clash_penalty_weight: float = 0.15
    graph_pair_clash_penalty_cap: float = 0.35
    graph_segment_link_penalty_weight: float = 0.12
    graph_segment_link_penalty_cap: float = 0.30
    graph_progress_penalty_weight: float = 0.10
    graph_progress_penalty_cap: float = 0.25
    graph_link_context_edge_penalty_weight: float = 0.50
    graph_link_context_edge_penalty_cap: float = 0.35
    auto_domain_residues: int = 120
    flexible: bool = True
    segment_shift_bound: float = 8.0
    segment_angle_bound_deg: float = 20.0
    flexible_refine_passes: int = 2
    flexible_seed_segments: int = 3
    flexible_seed_poses: int = 3
    flexible_bundle_limit: int = 6
    flexible_nleast: int = 5
    flexible_cutoff_score: float = -1.5
    flexible_progress_score_relax: float = 0.75
    flexible_screen_factor: int = 1
    growth_occupancy_penalty_weight: float = 12.0
    growth_restraint_power: float = 1.0
    growth_restraint_step_scale: float = 0.35
    growth_restraint_rscale: float = 15.0
    growth_pose_intensity_floor: float = 0.5
    growth_pose_intensity_cap: float = 4.0
    growth_pose_z_min_abs: float = 0.25
    flexible_revert_zcut: float = 0.5
    flexible_revert_min_std: float = 1.0
    segment_link_weight: float = 8.0
    segment_link_tolerance: float = 1.5
    max_graph_vertices: int = 256
    assembly_cycles: int = 8
    ldp_prune_exponent: float = 2.0
    ldp_min_keep_ratio: float = 0.05
    backend: str = "auto"
    device: str = "auto"
    grid_method: str = "ftmatch"
    max_shift_iterations: int = 256
    refine_method: str = "Powell"
    search_grid_output_path: str | None = None
    search_grid_ftmatch_output_path: str | None = None

    @property
    def ldp_kernel_scale(self) -> float:
        return 2.4 + 0.8 * self.resol
