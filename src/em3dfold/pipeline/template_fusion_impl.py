"""Run EM3DFold template-guided assembly on top of de novo models."""

import argparse
import builtins
import glob
import os
import shutil

from em3dfold.io.pdbio import chains_atom_pos_to_pdb, convert_to_chains, read_pdb
from em3dfold.template.pipeline import (
    assemble as template_assemble,
    denovo_fix_pipeline,
    denovo_imp_pipeline,
    preprocess,
)
from em3dfold.template.utils.misc_utils import abspath, pjoin


def print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    message = sep.join(str(arg) for arg in args)
    if not message.startswith("# "):
        message = f"# {message}"
    builtins.print(message, **kwargs)


def _default_template_lib_dir():
    return abspath(pjoin(os.path.dirname(__file__), ".."))


def _split_denovo_model_to_chains(model_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
        model_path,
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

    outputs = []
    for i in range(len(chains_atom_pos)):
        output_path = pjoin(out_dir, f"denovo_chain_{i}.pdb")
        chains_atom_pos_to_pdb(
            filename=output_path,
            chains_atom_pos=[chains_atom_pos[i]],
            chains_atom_mask=[chains_atom_mask[i]],
            chains_res_type=[chains_res_type[i]],
            chains_res_idx=[chains_res_idx[i]],
            chains_bfactor=[chains_bfactor[i]],
            suffix="pdb",
        )
        outputs.append(output_path)
    return outputs


def _existing_paths(paths):
    return [path for path in paths if path and os.path.exists(path)]


def _read_template_type(temp_dir: str) -> str:
    template_type_path = pjoin(temp_dir, "templs", "type.txt")
    if os.path.exists(template_type_path):
        with open(template_type_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return "none"


def _collect_assemble_inputs(fix_dir, imp_dir, denovo_model):
    candidates = [
        pjoin(imp_dir, "imp_chains_trimmed.cif"),
        pjoin(fix_dir, "fix_chains_templs.cif"),
        denovo_model,
    ]
    assemble_inputs = _existing_paths(candidates)
    if not assemble_inputs:
        raise RuntimeError("No structures were available for assembly.")

    print("# Assemble candidates (high to low priority):")
    for path in assemble_inputs:
        print(f"#   {path}")
    return assemble_inputs


def _write_final_outputs(final_output, output_dir):
    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
        final_output,
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

    output_path = pjoin(output_dir, "output.cif")
    chains_atom_pos_to_pdb(
        filename=output_path,
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_type=chains_res_type,
        chains_res_idx=chains_res_idx,
        chains_bfactor=chains_bfactor,
        suffix="cif",
    )

    legacy_output_path = pjoin(output_dir, "output_template_fused.cif")
    if legacy_output_path != output_path:
        shutil.copy(output_path, legacy_output_path)

    return output_path, legacy_output_path


def main(args):
    output_dir = abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = pjoin(output_dir, "template_pipeline")
    os.makedirs(temp_dir, exist_ok=True)

    lib_dir = abspath(args.lib_dir)
    denovo_model = abspath(args.denovo_model)
    map_path = abspath(args.map)
    seq_path = abspath(args.seq) if args.seq is not None else None

    if args.denovo_chain_dir is not None:
        denovo_chain_dir = abspath(args.denovo_chain_dir)
        denovo_chains = sorted(
            glob.glob(pjoin(denovo_chain_dir, "*.pdb"))
            + glob.glob(pjoin(denovo_chain_dir, "*.cif"))
        )
    else:
        denovo_chain_dir = pjoin(temp_dir, "denovo_chains")
        denovo_chains = _split_denovo_model_to_chains(denovo_model, denovo_chain_dir)

    print(f"# Using lib dir: {lib_dir}")
    print(f"# Using de novo model: {denovo_model}")
    print(f"# Found {len(denovo_chains)} de novo chain files")

    preprocess_args = argparse.Namespace(
        seq=seq_path,
        map=map_path,
        chain=args.template_chain,
        complex=args.template_complex,
        output=temp_dir,
        debug=args.debug,
    )
    preprocess.main(preprocess_args)

    template_type = _read_template_type(temp_dir)
    template_paths = sorted(glob.glob(pjoin(temp_dir, "templs", "templ_*.pdb")))
    has_templates = len(template_paths) > 0
    print(f"# Template type = {template_type}")
    print(f"# Prepared {len(template_paths)} template chains")

    fix_dir = pjoin(temp_dir, "fix")
    imp_dir = pjoin(temp_dir, "imp")
    assemble_dir = pjoin(temp_dir, "assemble")
    for path in [fix_dir, imp_dir, assemble_dir]:
        os.makedirs(path, exist_ok=True)

    if has_templates:
        print("# Run fix pipeline")
        fix_args = argparse.Namespace(
            seq=seq_path,
            chain=denovo_chains,
            template=template_paths,
            lib=lib_dir,
            output=fix_dir,
            verbose=args.verbose,
            debug=args.debug,
        )
        denovo_fix_pipeline.main(fix_args)

        print("# Run imp pipeline")
        imp_args = argparse.Namespace(
            seq=seq_path,
            chain=denovo_chains,
            template=template_paths,
            lib=lib_dir,
            output=imp_dir,
            verbose=args.verbose,
            debug=args.debug,
        )
        denovo_imp_pipeline.main(imp_args)
    else:
        print("# No template provided, assemble will fall back to de novo inputs")

    assemble_inputs = _collect_assemble_inputs(
        fix_dir=fix_dir,
        imp_dir=imp_dir,
        denovo_model=denovo_model,
    )

    print("# Run assemble stage")
    assemble_args = argparse.Namespace(
        seq=seq_path,
        pdb=assemble_inputs,
        map=map_path,
        verbose=args.verbose,
        lib=lib_dir,
        no_split=args.no_split,
        out=assemble_dir,
        debug=args.debug,
    )
    template_assemble.main(assemble_args)

    final_output = pjoin(assemble_dir, "assemble.cif")

    output_path, legacy_output_path = _write_final_outputs(final_output, output_dir)
    print(f"# Final assembled model written to {output_path}")
    print(f"# Legacy-compatible copy written to {legacy_output_path}")


def add_args(parser):
    parser.add_argument(
        "--denovo-model",
        required=True,
        help="EM3DFold de novo model (.cif/.pdb)",
    )
    parser.add_argument(
        "--denovo-chain-dir",
        default=None,
        help="Optional directory of per-chain de novo models",
    )
    parser.add_argument(
        "--map",
        required=True,
        help="Density map used for template fitting/assembly",
    )
    parser.add_argument(
        "--seq",
        default=None,
        help="Protein FASTA used by fix/imp steps",
    )
    parser.add_argument(
        "--template-chain",
        nargs="*",
        default=None,
        help="Single-chain protein template structures",
    )
    parser.add_argument(
        "--template-complex",
        default=None,
        help="Complex template structure",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--lib-dir",
        default=_default_template_lib_dir(),
        help="Directory containing template binaries under bin/",
    )
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Raise exceptions directly instead of falling back")
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the EM3DFold template-guided assembly pipeline.",
    )
    add_args(parser)
    main(parser.parse_args())
