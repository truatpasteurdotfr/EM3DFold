from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from em3dfit.assemble import assemble_chains
from em3dfit.config import Params
from em3dfit.meanshift import extract_ldps
from em3dfit.mrc import normalize_mrc, read_mrc, write_mrc
from em3dfit.pdbio import read_pdb, write_fitted_pdb, write_mcp_pdb, write_scored_pdb
from em3dfit.rigid import build_initial_ldp_search_grid, build_initial_ldp_search_grid_ftmatch
from em3dfit.score import score_chain_against_ldps
from em3dfit.log_utils import configure_runtime_logging, progress
from em3dfit.utils import (
    log_message as _base_log_message,
    normalize_device_spec,
    stage_timer as _base_stage_timer,
)

LOG_STAGE = "Cli"
log_message = partial(_base_log_message, stage=LOG_STAGE)
stage_timer = partial(_base_stage_timer, stage=LOG_STAGE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="em3dfit",
        description="Python refactor of the EMBuild MCP extraction, scoring, and assembly pipeline.",
    )
    parser.add_argument("main_chain_map")
    parser.add_argument("input_pdb")
    parser.add_argument("resolution", type=float)
    parser.add_argument("output_pdb")
    parser.add_argument("-apix", type=float, default=1.0)
    parser.add_argument("-thresh", dest="threshold", type=float, default=20.0)
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.10,
        help="Set threshold to threshold_ratio * max(map_value) after map normalization",
    )
    parser.add_argument("-rshift", type=float, default=10.0)
    parser.add_argument("-rmerge", type=float, default=1.0)
    parser.add_argument("-filter", dest="filter_fraction", type=float, default=0.03)
    parser.add_argument("-angle_step", type=float, default=18.0)
    parser.add_argument("-fgrid", type=float, default=3.0)
    parser.add_argument("-sgrid", type=float, default=2.0)
    parser.add_argument("-rmsdcut1", type=float, default=2.5)
    parser.add_argument("-rmsdcut2", type=float, default=5.0)
    parser.add_argument("-ntop", type=int, default=10)
    parser.add_argument("--rigid-nleast", type=int, default=5)
    parser.add_argument("--rigid-cutoff-score-early", type=float, default=-1.5)
    parser.add_argument("--rigid-cutoff-score-late", type=float, default=-0.5)
    parser.add_argument("--rigid-skip-short-residues", type=int, default=50)
    parser.add_argument("-ntrans", type=int, default=8)
    parser.add_argument("--assembly-cycles", type=int, default=10)
    parser.add_argument("--auto-domain-residues", type=int, default=120)
    parser.add_argument("--flexible-nleast", type=int, default=5)
    parser.add_argument("--flexible-cutoff-score", type=float, default=-1.5)
    parser.add_argument("--external-clash-pair-cutoff", type=float, default=0.15)
    parser.add_argument("--external-clash-cutoff", type=float, default=0.30)
    parser.add_argument("--external-clash-penalty-weight", type=float, default=2.0)
    parser.add_argument("--external-clash-score-cap", type=float, default=0.95)
    parser.add_argument("--ldp-coverage-reward-weight", type=float, default=0.20)
    parser.add_argument("--graph-external-clash-max-penalty-weight", type=float, default=0.10)
    parser.add_argument("--graph-external-clash-max-penalty-cap", type=float, default=0.25)
    parser.add_argument("--graph-pair-clash-penalty-weight", type=float, default=0.15)
    parser.add_argument("--graph-pair-clash-penalty-cap", type=float, default=0.35)
    parser.add_argument("--graph-segment-link-penalty-weight", type=float, default=0.12)
    parser.add_argument("--graph-segment-link-penalty-cap", type=float, default=0.30)
    parser.add_argument("--graph-progress-penalty-weight", type=float, default=0.10)
    parser.add_argument("--graph-progress-penalty-cap", type=float, default=0.25)
    parser.add_argument("--graph-link-context-edge-penalty-weight", type=float, default=0.50)
    parser.add_argument("--graph-link-context-edge-penalty-cap", type=float, default=0.35)
    parser.add_argument("--max-graph-vertices", type=int, default=256)
    parser.add_argument("--no-flexible", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--mcp", action="store_true")
    parser.add_argument("--rigid", action="store_true", help="Run rigid fitting and greedy assembly.")
    parser.add_argument("--backend", choices=["auto", "torch", "scipy"], default="auto")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selector: auto, cpu, cuda, cuda:N, or a bare GPU index such as 0",
    )
    parser.add_argument("--grid-method", choices=["ftmatch", "smoothed"], default="ftmatch")
    parser.add_argument("--refine-method", choices=["Powell", "Nelder-Mead"], default="Powell")
    parser.add_argument("--write-search-grid-mrc")
    parser.add_argument("--write-search-grid-mrc-ftmatch")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_runtime_logging(
        Path(args.output_pdb).expanduser().resolve().parent,
        package_prefixes=("em3dfit",),
        progress_logger_name="em3dfit.progress",
        helper_modules=("em3dfit.log_utils", "em3dfit.utils"),
    )
    device = normalize_device_spec(args.device)

    params = Params(
        resol=args.resolution,
        apix=args.apix,
        threshold=args.threshold,
        rshift=args.rshift,
        rmerge=args.rmerge,
        filter_fraction=args.filter_fraction,
        angle_step=args.angle_step,
        fgrid=args.fgrid,
        sgrid=args.sgrid,
        rmsdcut1=args.rmsdcut1,
        rmsdcut2=args.rmsdcut2,
        ntop=args.ntop,
        rigid_nleast=args.rigid_nleast,
        rigid_cutoff_score_early=args.rigid_cutoff_score_early,
        rigid_cutoff_score_late=args.rigid_cutoff_score_late,
        rigid_skip_short_residues=args.rigid_skip_short_residues,
        ntrans=args.ntrans,
        external_clash_pair_cutoff=args.external_clash_pair_cutoff,
        external_clash_cutoff=args.external_clash_cutoff,
        external_clash_penalty_weight=args.external_clash_penalty_weight,
        external_clash_score_cap=args.external_clash_score_cap,
        ldp_coverage_reward_weight=args.ldp_coverage_reward_weight,
        graph_external_clash_max_penalty_weight=args.graph_external_clash_max_penalty_weight,
        graph_external_clash_max_penalty_cap=args.graph_external_clash_max_penalty_cap,
        graph_pair_clash_penalty_weight=args.graph_pair_clash_penalty_weight,
        graph_pair_clash_penalty_cap=args.graph_pair_clash_penalty_cap,
        graph_segment_link_penalty_weight=args.graph_segment_link_penalty_weight,
        graph_segment_link_penalty_cap=args.graph_segment_link_penalty_cap,
        graph_progress_penalty_weight=args.graph_progress_penalty_weight,
        graph_progress_penalty_cap=args.graph_progress_penalty_cap,
        graph_link_context_edge_penalty_weight=args.graph_link_context_edge_penalty_weight,
        graph_link_context_edge_penalty_cap=args.graph_link_context_edge_penalty_cap,
        max_graph_vertices=args.max_graph_vertices,
        assembly_cycles=args.assembly_cycles,
        auto_domain_residues=args.auto_domain_residues,
        flexible_nleast=args.flexible_nleast,
        flexible_cutoff_score=args.flexible_cutoff_score,
        flexible=not (args.no_flexible or args.rigid),
        backend=args.backend,
        device=device,
        grid_method=args.grid_method,
        refine_method=args.refine_method,
        search_grid_output_path=args.write_search_grid_mrc,
        search_grid_ftmatch_output_path=args.write_search_grid_mrc_ftmatch,
    )

    progress("Read density map")
    log_message(f"reading map {args.main_chain_map}")
    with stage_timer("read mrc"):
        raw_map = read_mrc(args.main_chain_map)
    with stage_timer("normalize mrc"):
        norm_map = normalize_mrc(raw_map, params.apix)
    log_message(f"normalized map shape {norm_map.data.shape}, voxel size {np.asarray(norm_map.voxel_size)}")
    max_density = float(np.max(norm_map.data))
    if args.threshold_ratio is not None:
        params.threshold = float(args.threshold_ratio) * max_density
        log_message(
            f"threshold-ratio {float(args.threshold_ratio):.3f} -> "
            f"threshold {params.threshold:.3f} from map max {max_density:.3f}"
        )
    elif params.threshold >= max_density:
        adaptive = float(np.percentile(norm_map.data, 99.0))
        log_message(
            f"threshold {params.threshold:.3f} is above map max {max_density:.3f}; "
            f"falling back to p99={adaptive:.3f} for this density map"
        )
        params.threshold = adaptive

    progress("Read input structure")
    log_message(f"reading pdb {args.input_pdb}")
    with stage_timer("read pdb"):
        model = read_pdb(args.input_pdb)
    log_message(f"chains {len(model.chains)}, residues {model.total_residues}, torch cuda available {torch.cuda.is_available()}")
    log_message(
        "graph tuning "
        f"ext_pair_cutoff={params.external_clash_pair_cutoff:.3f}, "
        f"ext_sum_cutoff={params.external_clash_cutoff:.3f}, "
        f"ext_penalty_weight={params.external_clash_penalty_weight:.3f}, "
        f"coverage_reward_weight={params.ldp_coverage_reward_weight:.3f}, "
        f"graph_ext_max_penalty_weight={params.graph_external_clash_max_penalty_weight:.3f}, "
        f"graph_pair_penalty_weight={params.graph_pair_clash_penalty_weight:.3f}, "
        f"graph_segment_link_penalty_weight={params.graph_segment_link_penalty_weight:.3f}, "
        f"graph_progress_penalty_weight={params.graph_progress_penalty_weight:.3f}, "
        f"graph_link_context_edge_penalty_weight={params.graph_link_context_edge_penalty_weight:.3f}, "
        f"max_graph_vertices={params.max_graph_vertices}"
    )

    progress("Extract LDPs")
    log_message("extracting MCP/LDP points")
    with stage_timer("extract ldps"):
        ldps, ldps_dens, _membership = extract_ldps(norm_map, params)
    log_message(f"main-chain points {len(ldps)}")
    if params.search_grid_output_path:
        search_grid_path = Path(params.search_grid_output_path)
        with stage_timer("write search grid mrc"):
            search_grid, search_origin = build_initial_ldp_search_grid(ldps, ldps_dens, params)
            write_mrc(search_grid_path, search_grid, voxel_size=params.fgrid, origin=search_origin)
        log_message(f"write initial search grid MRC to {search_grid_path} method {params.grid_method}")
    if params.search_grid_ftmatch_output_path:
        search_grid_ftmatch_path = Path(params.search_grid_ftmatch_output_path)
        with stage_timer("write ftmatch search grid mrc"):
            search_grid_ftmatch, search_origin_ftmatch, search_backend = build_initial_ldp_search_grid_ftmatch(
                ldps,
                ldps_dens,
                params,
            )
            write_mrc(search_grid_ftmatch_path, search_grid_ftmatch, voxel_size=params.fgrid, origin=search_origin_ftmatch)
        log_message(f"write ftmatch-style initial search grid MRC to {search_grid_ftmatch_path} backend {search_backend}")

    output_path = Path(args.output_pdb)
    if args.mcp:
        progress("Write MCP structure")
        with stage_timer("write mcp pdb"):
            write_mcp_pdb(output_path, ldps, ldps_dens)
        log_message(f"write MCP structure to {output_path}")
        return 0

    if args.score:
        progress("Score chains")
        with stage_timer("score chains"):
            for chain in model.chains:
                chain.solutions = np.zeros((chain.n_segments, 6), dtype=np.float32)
                chain.scores = score_chain_against_ldps(chain, ldps, ldps_dens, params)
        with stage_timer("write scored pdb"):
            write_scored_pdb(model, output_path)
        log_message(f"write scored structure to {output_path}")
        return 0

    progress("Run EM3DFit assembly")
    log_message("running rigid/flexible assembly")
    with stage_timer("assemble chains"):
        pose_sets = assemble_chains(model.chains, ldps, ldps_dens, params)
    for chain_idx, poses in enumerate(pose_sets, start=1):
        if not poses:
            log_message(f"chain {chain_idx:02d} no candidate poses were retained")
            continue
        if model.chains[chain_idx - 1].solutions is None:
            log_message(
                f"chain {chain_idx:02d} unresolved after assembly, "
                f"kept {len(poses)} rigid pose(s), best score {poses[0].score:.3f}"
            )
            continue
        best = poses[0]
        log_message(
            f"chain {chain_idx:02d} segments {model.chains[chain_idx - 1].n_segments}, "
            f"kept {len(poses)} rigid pose(s), best score {best.score:.3f}"
        )
    with stage_timer("write fitted pdb"):
        write_fitted_pdb(model, output_path)
    log_message(f"write fitted structure to {output_path}")
    progress(f"Write fitted structure to {output_path}")
    return 0
