"""EM3DFold template fit pipeline built on EM3DFit.

This module is an EM3DFold-side pipeline entrypoint. It prepares template
chains/domains and then calls the actual rigid-fitting/assembly implementation
from the standalone ``em3dfit`` package under ``pkgs/EM3DFit``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from em3dfold.io.pdbio import chains_atom_pos_to_pdb, read_pdb
from em3dfold.pipeline.unidoc_domain import parse_unidoc_domains
from em3dfold.utils.misc_utils import abspath, pjoin
from em3dfold.utils.torch_utils import seed_everything


DEFAULT_RANDOM_SEED = 42


def _ensure_local_em3dfit_on_path():
    repo_root = Path(__file__).resolve().parents[3]
    em3dfit_src = repo_root / "pkgs" / "EM3DFit" / "src"
    if em3dfit_src.exists():
        em3dfit_src_str = str(em3dfit_src)
        if em3dfit_src_str not in sys.path:
            sys.path.insert(0, em3dfit_src_str)


def _load_em3dfit_modules():
    # This file is only an interface layer. The actual fitting logic lives in
    # the standalone em3dfit package and is imported lazily here.
    _ensure_local_em3dfit_on_path()
    from em3dfit.assemble import assemble_chains
    from em3dfit.config import Params
    from em3dfit.meanshift import extract_ldps
    from em3dfit.mrc import normalize_mrc, read_mrc
    from em3dfit.pdbio import read_pdb as read_em3dfit_pdb
    from em3dfit.pdbio import write_mcp_pdb
    from em3dfit.pdbio import write_fitted_pdb

    return {
        "assemble_chains": assemble_chains,
        "Params": Params,
        "extract_ldps": extract_ldps,
        "normalize_mrc": normalize_mrc,
        "read_mrc": read_mrc,
        "read_em3dfit_pdb": read_em3dfit_pdb,
        "write_mcp_pdb": write_mcp_pdb,
        "write_fitted_pdb": write_fitted_pdb,
    }


def _resolve_em3dfit_backend_and_device(device):
    device = str(device).strip().lower()
    if device in {"", "cpu"}:
        return "scipy", "cpu"
    if device == "auto":
        try:
            import torch
        except ImportError:
            return "scipy", "cpu"
        if torch.cuda.is_available():
            return "torch", "cuda"
        return "scipy", "cpu"
    if device.isdigit():
        return "torch", f"cuda:{device}"
    if device == "cuda" or device.startswith("cuda:"):
        return "torch", device
    return "scipy", "cpu"


def _parse_domain_string(domain_str):
    ranges = []
    for field in str(domain_str).split(","):
        field = field.strip()
        if not field:
            continue
        start_text, end_text = field.split("~", 1)
        ranges.append((int(start_text), int(end_text)))
    return ranges


def _mask_for_domain(res_idx, domain_str):
    mask = np.zeros(len(res_idx), dtype=bool)
    for start, end in _parse_domain_string(domain_str):
        mask |= (res_idx >= start) & (res_idx <= end)
    return mask


def _write_chain_structure(
    output_path,
    atom_pos,
    atom_mask,
    res_type,
    res_idx,
    bfactor,
    *,
    suffix,
):
    chains_atom_pos_to_pdb(
        output_path,
        chains_atom_pos=[atom_pos],
        chains_atom_mask=[atom_mask],
        chains_res_type=[res_type],
        chains_res_idx=[res_idx],
        chains_idx=[0],
        chains_bfactor=[bfactor],
        suffix=suffix,
    )


def _extract_protein_template_chains(template_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    chain_records = []

    for template_idx, template_path in enumerate(template_paths):
        template_path = abspath(template_path)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Protein template file not found: {template_path}")

        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
            template_path,
            keep_valid=False,
            return_bfactor=True,
        )

        protein_mask = res_type < 20
        if not np.any(protein_mask):
            print(f"# No protein residues found in template {template_path}, skip")
            continue

        atom_pos = atom_pos[protein_mask]
        atom_mask = atom_mask[protein_mask]
        res_type = res_type[protein_mask]
        res_idx = res_idx[protein_mask]
        chain_idx = chain_idx[protein_mask]
        bfactor = bfactor[protein_mask]

        unique_chain_indices = np.unique(chain_idx)
        for chain_local_idx, source_chain_idx in enumerate(unique_chain_indices):
            chain_mask = chain_idx == source_chain_idx
            output_path = pjoin(
                output_dir,
                f"template_{template_idx}_chain_{chain_local_idx}.cif",
            )
            _write_chain_structure(
                output_path,
                atom_pos[chain_mask],
                atom_mask[chain_mask],
                res_type[chain_mask],
                res_idx[chain_mask],
                bfactor[chain_mask],
                suffix="cif",
            )
            print(f"# Write protein template chain to {output_path}")
            chain_records.append(
                {
                    "template_index": template_idx,
                    "template_path": template_path,
                    "source_chain_index": int(source_chain_idx),
                    "chain_local_index": chain_local_idx,
                    "chain_path": output_path,
                }
            )

    return chain_records


def _split_chain_to_domains(chain_record, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    chain_path = chain_record["chain_path"]
    domain_result = parse_unidoc_domains(chain_path, chain_id="A")["A"]
    domain_strings = list(domain_result.get("large_domain_list", []))
    if not domain_strings:
        atom_pos, _atom_mask, _res_type, res_idx, _chain_idx = read_pdb(chain_path, keep_valid=False)
        domain_strings = [f"{int(np.min(res_idx))}~{int(np.max(res_idx))}"]

    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
        chain_path,
        keep_valid=False,
        return_bfactor=True,
    )
    if np.any(chain_idx != 0):
        raise ValueError(f"Expected a single-chain template at {chain_path}")

    domain_records = []
    for domain_idx, domain_str in enumerate(domain_strings):
        domain_mask = _mask_for_domain(res_idx, domain_str)
        if not np.any(domain_mask):
            raise ValueError(f"Domain {domain_str} selects no residues in {chain_path}")

        output_path = pjoin(
            output_dir,
            (
                f"template_{chain_record['template_index']}_"
                f"chain_{chain_record['chain_local_index']}_"
                f"domain_{domain_idx}.cif"
            ),
        )
        _write_chain_structure(
            output_path,
            atom_pos[domain_mask],
            atom_mask[domain_mask],
            res_type[domain_mask],
            res_idx[domain_mask],
            bfactor[domain_mask],
            suffix="cif",
        )
        print(f"# Write domain {domain_idx} to {output_path}")
        domain_records.append(
            {
                **chain_record,
                "domain_index": domain_idx,
                "domain_string": domain_str,
                "domain_path": output_path,
            }
        )

    return domain_result, domain_records


def _infer_map_apix_and_normalize(raw_map, normalize_mrc):
    voxel_size = np.asarray(raw_map.voxel_size, dtype=np.float32)
    if voxel_size.shape != (3,) or np.any(~np.isfinite(voxel_size)) or np.any(voxel_size <= 0):
        raise ValueError(f"Invalid voxel size in map header: {voxel_size}")

    inferred_apix = float(np.mean(voxel_size))
    if np.allclose(voxel_size, inferred_apix, rtol=1e-3, atol=1e-4):
        print(
            "# EM3DFit inferred apix = {:.4f} A from map header; keep native grid".format(
                inferred_apix
            )
        )
        return inferred_apix, normalize_mrc(raw_map, -1.0)

    print(
        "# EM3DFit detected anisotropic voxel size {} ; resample to inferred isotropic apix = {:.4f} A".format(
            [float(x) for x in voxel_size],
            inferred_apix,
        )
    )
    return inferred_apix, normalize_mrc(raw_map, inferred_apix)


def _prepare_ldps(
    map_path,
    resolution,
    threshold,
    threshold_ratio,
    device,
    angle_step,
    fgrid,
    sgrid,
    ntrans,
    ntop,
    rshift,
    rmerge,
    rmsdcut1,
    rmsdcut2,
    rigid_nleast,
    rigid_cutoff_score_early,
    rigid_cutoff_score_late,
    rigid_skip_short_residues,
):
    modules = _load_em3dfit_modules()
    Params = modules["Params"]
    read_mrc = modules["read_mrc"]
    normalize_mrc = modules["normalize_mrc"]
    extract_ldps = modules["extract_ldps"]

    raw_map = read_mrc(map_path)
    inferred_apix, norm_map = _infer_map_apix_and_normalize(raw_map, normalize_mrc)
    backend, resolved_device = _resolve_em3dfit_backend_and_device(device)
    params = Params(
        resol=resolution,
        apix=inferred_apix,
        threshold=threshold,
        rshift=rshift,
        rmerge=rmerge,
        angle_step=angle_step,
        fgrid=fgrid,
        sgrid=sgrid,
        rmsdcut1=rmsdcut1,
        rmsdcut2=rmsdcut2,
        ntrans=ntrans,
        ntop=ntop,
        rigid_nleast=rigid_nleast,
        rigid_cutoff_score_early=rigid_cutoff_score_early,
        rigid_cutoff_score_late=rigid_cutoff_score_late,
        rigid_skip_short_residues=rigid_skip_short_residues,
        flexible=False,
        backend=backend,
        device=resolved_device,
        grid_method="ftmatch",
    )

    max_density = float(np.max(norm_map.data))
    if threshold_ratio is not None:
        params.threshold = float(threshold_ratio) * max_density
        print(
            "# Fit threshold-ratio {:.3f} -> threshold {:.3f} from map max {:.3f}".format(
                float(threshold_ratio),
                params.threshold,
                max_density,
            )
        )
    elif params.threshold >= max_density:
        params.threshold = float(np.percentile(norm_map.data, 99.0))
        print(
            "# Fit threshold is above map max, fallback to p99 = {:.3f}".format(
                params.threshold
            )
        )

    ldps, ldps_dens, _membership = extract_ldps(norm_map, params)
    print(f"# EM3DFit backend = {backend}, device = {resolved_device}")
    print(f"# Extracted {len(ldps)} LDPs from density map")
    return params, ldps, ldps_dens


def _write_domain_group_pdb(domain_records, output_path):
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_type = []
    chains_res_idx = []
    chains_bfactor = []

    for domain_record in domain_records:
        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
            domain_record["domain_path"],
            keep_valid=False,
            return_bfactor=True,
        )
        if np.any(chain_idx != 0):
            raise ValueError(
                f"Expected a single-chain domain PDB at {domain_record['domain_path']}"
            )
        chains_atom_pos.append(atom_pos)
        chains_atom_mask.append(atom_mask)
        chains_res_type.append(res_type)
        chains_res_idx.append(res_idx)
        chains_bfactor.append(bfactor)

    chains_atom_pos_to_pdb(
        output_path,
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_type=chains_res_type,
        chains_res_idx=chains_res_idx,
        chains_idx=list(range(len(domain_records))),
        chains_bfactor=chains_bfactor,
        suffix=Path(output_path).suffix.lstrip(".") or "cif",
    )
    print(f"# Write grouped domains to {output_path}")


def _fit_domain_group(
    domain_records,
    grouped_input_path,
    grouped_fitted_path,
    fitted_dir,
    ldps,
    ldps_dens,
    params,
):
    modules = _load_em3dfit_modules()
    assemble_chains = modules["assemble_chains"]
    read_em3dfit_pdb = modules["read_em3dfit_pdb"]
    write_fitted_pdb = modules["write_fitted_pdb"]

    model = read_em3dfit_pdb(grouped_input_path)
    if len(model.chains) != len(domain_records):
        raise ValueError(
            "Grouped EM3DFit input chain count does not match domain count: "
            f"{len(model.chains)} vs {len(domain_records)}"
        )

    pose_sets = assemble_chains(model.chains, ldps, ldps_dens, params)
    if any(chain.solutions is not None for chain in model.chains):
        write_fitted_pdb(model, grouped_fitted_path)
        print(f"# Write grouped fitted domains to {grouped_fitted_path}")
    else:
        grouped_fitted_path = None
        print("# No grouped fitted domains were resolved; skip total CIF output")

    fit_results = []
    for domain_record, chain, poses in zip(domain_records, model.chains, pose_sets, strict=True):
        fitted_path = pjoin(
            fitted_dir,
            (
                f"template_{domain_record['template_index']}_"
                f"chain_{domain_record['chain_local_index']}_"
                f"domain_{domain_record['domain_index']}_fitted.cif"
            ),
        )

        if chain.solutions is None:
            fit_results.append(
                {
                    "fitted_path": None,
                    "score": None,
                    "solution": None,
                    "resolved": False,
                }
            )
            print(
                "# Domain unresolved in grouped fitting, skip fitted output for {}".format(
                    domain_record["domain_path"]
                )
            )
            continue

        domain_model = read_em3dfit_pdb(domain_record["domain_path"])
        domain_model.chains[0].solutions = np.asarray(chain.solutions, dtype=np.float32).copy()
        write_fitted_pdb(domain_model, fitted_path)
        best_pose = poses[0] if poses else None
        fit_results.append(
            {
                "fitted_path": fitted_path,
                "score": None if best_pose is None else float(best_pose.score),
                "solution": None
                if best_pose is None
                else np.asarray(best_pose.solution, dtype=np.float32).tolist(),
                "resolved": True,
            }
        )
        print(f"# Write fitted domain to {fitted_path}")

    return fit_results, grouped_fitted_path


def run_fit_pipeline(
    template_paths,
    map_path,
    output_dir,
    *,
    resolution=5.0,
    threshold=20.0,
    threshold_ratio=None,
    device="auto",
    angle_step=18.0,
    fgrid=3.0,
    sgrid=2.0,
    ntrans=8,
    ntop=10,
    rshift=1.0,
    rmerge=1.0,
    rmsdcut1=2.5,
    rmsdcut2=5.0,
    rigid_nleast=5,
    rigid_cutoff_score_early=-1.5,
    rigid_cutoff_score_late=-0.5,
    rigid_skip_short_residues=50,
):
    seed_everything(DEFAULT_RANDOM_SEED)
    output_dir = abspath(output_dir)
    map_path = abspath(map_path)
    os.makedirs(output_dir, exist_ok=True)

    chain_dir = pjoin(output_dir, "chains")
    domain_dir = pjoin(output_dir, "domains")
    fitted_dir = pjoin(output_dir, "fitted_domains")
    os.makedirs(chain_dir, exist_ok=True)
    os.makedirs(domain_dir, exist_ok=True)
    os.makedirs(fitted_dir, exist_ok=True)

    params, ldps, ldps_dens = _prepare_ldps(
        map_path=map_path,
        resolution=resolution,
        threshold=threshold,
        threshold_ratio=threshold_ratio,
        device=device,
        angle_step=angle_step,
        fgrid=fgrid,
        sgrid=sgrid,
        ntrans=ntrans,
        ntop=ntop,
        rshift=rshift,
        rmerge=rmerge,
        rmsdcut1=rmsdcut1,
        rmsdcut2=rmsdcut2,
        rigid_nleast=rigid_nleast,
        rigid_cutoff_score_early=rigid_cutoff_score_early,
        rigid_cutoff_score_late=rigid_cutoff_score_late,
        rigid_skip_short_residues=rigid_skip_short_residues,
    )
    modules = _load_em3dfit_modules()
    initial_ldps_path = pjoin(output_dir, "initial_ldps.cif")
    modules["write_mcp_pdb"](initial_ldps_path, ldps, ldps_dens)
    print(f"# Write initial LDPs to {initial_ldps_path}")

    chain_records = _extract_protein_template_chains(template_paths, chain_dir)
    summary = {
        "map_path": map_path,
        "output_dir": output_dir,
        "resolution": resolution,
        "apix": float(params.apix),
        "threshold": float(params.threshold),
        "threshold_ratio": None if threshold_ratio is None else float(threshold_ratio),
        "rshift": float(params.rshift),
        "rmerge": float(params.rmerge),
        "rmsdcut1": float(params.rmsdcut1),
        "rmsdcut2": float(params.rmsdcut2),
        "rigid_nleast": int(params.rigid_nleast),
        "rigid_cutoff_score_early": float(params.rigid_cutoff_score_early),
        "rigid_cutoff_score_late": float(params.rigid_cutoff_score_late),
        "rigid_skip_short_residues": int(params.rigid_skip_short_residues),
        "initial_ldps_path": initial_ldps_path,
        "grouped_input_path": None,
        "grouped_fitted_path": None,
        "fitted_total_path": None,
        "chains": [],
    }

    all_domain_records = []
    chain_summaries = []
    for chain_record in chain_records:
        domain_result, domain_records = _split_chain_to_domains(chain_record, domain_dir)
        chain_summary = {
            "template_index": chain_record["template_index"],
            "template_path": chain_record["template_path"],
            "source_chain_index": chain_record["source_chain_index"],
            "chain_local_index": chain_record["chain_local_index"],
            "chain_path": chain_record["chain_path"],
            "fragment_list": domain_result.get("fragment_list", []),
            "small_domain_list": domain_result.get("small_domain_list", []),
            "large_domain_list": domain_result.get("large_domain_list", []),
            "grouped_input_path": None,
            "grouped_fitted_path": None,
            "domains": [],
        }
        for domain_record in domain_records:
            chain_summary["domains"].append(
                {
                    "domain_index": domain_record["domain_index"],
                    "domain_string": domain_record["domain_string"],
                    "domain_path": domain_record["domain_path"],
                    "fitted_path": None,
                    "score": None,
                    "solution": None,
                    "resolved": False,
                }
            )
        chain_summaries.append(chain_summary)
        all_domain_records.extend(domain_records)

    if all_domain_records:
        grouped_input_path = pjoin(domain_dir, "all_domains_grouped.cif")
        grouped_fitted_path = pjoin(output_dir, "fitted_total.cif")
        summary["grouped_input_path"] = grouped_input_path
        summary["grouped_fitted_path"] = grouped_fitted_path
        summary["fitted_total_path"] = grouped_fitted_path

        _write_domain_group_pdb(all_domain_records, grouped_input_path)
        fit_results, resolved_grouped_fitted_path = _fit_domain_group(
            all_domain_records,
            grouped_input_path,
            grouped_fitted_path,
            fitted_dir,
            ldps,
            ldps_dens,
            params,
        )
        summary["grouped_fitted_path"] = resolved_grouped_fitted_path
        summary["fitted_total_path"] = resolved_grouped_fitted_path

        fit_result_by_key = {}
        for domain_record, fit_result in zip(all_domain_records, fit_results, strict=True):
            record_key = (
                domain_record["template_index"],
                domain_record["chain_local_index"],
                domain_record["domain_index"],
            )
            fit_result_by_key[record_key] = fit_result

        for chain_summary in chain_summaries:
            chain_summary["grouped_input_path"] = grouped_input_path
            chain_summary["grouped_fitted_path"] = resolved_grouped_fitted_path
            for domain_summary in chain_summary["domains"]:
                record_key = (
                    chain_summary["template_index"],
                    chain_summary["chain_local_index"],
                    domain_summary["domain_index"],
                )
                domain_summary.update(fit_result_by_key[record_key])

    summary["chains"] = chain_summaries

    summary_path = pjoin(output_dir, "fit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"# Write fit summary to {summary_path}")
    return summary


def add_args(parser):
    parser.add_argument("--protein-template", "-pt", nargs="+", required=True)
    parser.add_argument("--map", "-m", required=True, help="Input density map")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--resolution", type=float, default=5.0, help="Map resolution for EM3DFit")
    parser.add_argument("--threshold", type=float, default=20.0, help="Density threshold for LDP extraction")
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=None,
        help="If set, use threshold_ratio * max(map_value) instead of a fixed threshold",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="If GPU is requested (e.g. 0 or cuda), use torch+cuda; otherwise use scipy backend.",
    )
    parser.add_argument("--angle-step", type=float, default=18.0, help="Angular step for rigid search")
    parser.add_argument("--fgrid", type=float, default=3.0, help="Coarse rigid-search grid spacing")
    parser.add_argument("--sgrid", type=float, default=2.0, help="Refinement grid spacing")
    parser.add_argument("--ntrans", type=int, default=8, help="Top translations kept per rotation")
    parser.add_argument("--ntop", type=int, default=10, help="Top rigid poses kept per domain")
    parser.add_argument("--rshift", type=float, default=1.0, help="Mean-shift maximum travel distance in Angstrom")
    parser.add_argument("--rmerge", type=float, default=1.0, help="LDP mode merge radius in Angstrom")
    parser.add_argument("--rmsdcut1", type=float, default=2.5, help="Rigid pose clustering RMSD cutoff")
    parser.add_argument("--rmsdcut2", type=float, default=5.0, help="Reserved secondary RMSD cutoff passed to EM3DFit")
    parser.add_argument("--rigid-nleast", type=int, default=5, help="Minimum rigid poses retained after z-score filtering")
    parser.add_argument(
        "--rigid-cutoff-score-early",
        type=float,
        default=-1.5,
        help="Early rigid z-score cutoff passed to EM3DFit",
    )
    parser.add_argument(
        "--rigid-cutoff-score-late",
        type=float,
        default=-0.5,
        help="Late rigid z-score cutoff passed to EM3DFit search",
    )
    parser.add_argument(
        "--rigid-skip-short-residues",
        type=int,
        default=50,
        help="Rigid fitting short-chain skip threshold passed to EM3DFit",
    )
    return parser


def main(args):
    return run_fit_pipeline(
        template_paths=args.protein_template,
        map_path=args.map,
        output_dir=args.output,
        resolution=args.resolution,
        threshold=args.threshold,
        threshold_ratio=getattr(args, "threshold_ratio", None),
        device=args.device,
        angle_step=args.angle_step,
        fgrid=args.fgrid,
        sgrid=args.sgrid,
        ntrans=args.ntrans,
        ntop=args.ntop,
        rshift=getattr(args, "rshift", 1.0),
        rmerge=getattr(args, "rmerge", 1.0),
        rmsdcut1=getattr(args, "rmsdcut1", 2.5),
        rmsdcut2=getattr(args, "rmsdcut2", 5.0),
        rigid_nleast=getattr(args, "rigid_nleast", 5),
        rigid_cutoff_score_early=getattr(args, "rigid_cutoff_score_early", -1.5),
        rigid_cutoff_score_late=getattr(args, "rigid_cutoff_score_late", -0.5),
        rigid_skip_short_residues=getattr(args, "rigid_skip_short_residues", 50),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "EM3DFold interface for EM3DFit: split protein templates into "
            "chains/domains, then call the standalone EM3DFit package for "
            "group rigid fitting."
        ),
    )
    main(add_args(parser).parse_args())
