#!/usr/bin/env python3
"""Split one structure file into combined protein and NA CIF/PDB files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser


def _ensure_repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_ensure_repo_src_on_path()
from em3dfold.io.pdbio import chain_names, chains_atom_pos_to_pdb, convert_to_chains, read_pdb  # noqa: E402


PDB_SUFFIXES = {".pdb", ".ent"}
CIF_SUFFIXES = {".cif", ".mmcif"}


def _build_parser_for_path(path: Path):
    suffix = path.suffix.lower()
    if suffix in PDB_SUFFIXES:
        return PDBParser(QUIET=True)
    if suffix in CIF_SUFFIXES:
        return MMCIFParser(QUIET=True)
    raise ValueError(f"Unsupported structure format: {path}")


def _collect_original_chain_ids(structure_path: Path) -> list[str]:
    parser = _build_parser_for_path(structure_path)
    structure = parser.get_structure(structure_path.stem or "model", str(structure_path))
    model = structure[0]
    return [str(chain.id).strip() for chain in model]


def _resolve_writer_chain_index(chain_id: str, fallback_index: int) -> int:
    normalized = str(chain_id).strip()
    if normalized in chain_names:
        return chain_names.index(normalized)
    return fallback_index


def _rewrite_chain_ids_in_cif(output_path: Path, chain_id_map: dict[str, str]) -> None:
    if not chain_id_map:
        return

    lines = output_path.read_text().splitlines()
    header_indices: dict[str, int] = {}
    atom_site_header_order: list[str] = []
    in_atom_site_loop = False
    data_start = -1

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_site_loop = False
            atom_site_header_order = []
            header_indices = {}
            continue
        if stripped.startswith("_atom_site."):
            if not in_atom_site_loop:
                in_atom_site_loop = True
            atom_site_header_order.append(stripped)
            header_indices[stripped] = len(atom_site_header_order) - 1
            continue
        if in_atom_site_loop and stripped and not stripped.startswith("_") and not stripped.startswith("#"):
            data_start = idx
            break

    if data_start < 0:
        return

    label_idx = header_indices.get("_atom_site.label_asym_id")
    auth_idx = header_indices.get("_atom_site.auth_asym_id")
    if label_idx is None and auth_idx is None:
        return

    for idx in range(data_start, len(lines)):
        stripped = lines[idx].strip()
        if not stripped or stripped == "#":
            break
        tokens = stripped.split()
        if label_idx is not None and label_idx < len(tokens):
            tokens[label_idx] = chain_id_map.get(tokens[label_idx], tokens[label_idx])
        if auth_idx is not None and auth_idx < len(tokens):
            tokens[auth_idx] = chain_id_map.get(tokens[auth_idx], tokens[auth_idx])
        lines[idx] = " ".join(tokens)

    output_path.write_text("\n".join(lines) + "\n")


def _write_subset(
    output_path: Path,
    *,
    chains_atom_pos: list[np.ndarray],
    chains_atom_mask: list[np.ndarray],
    chains_res_type: list[np.ndarray],
    chains_res_idx: list[np.ndarray],
    chains_bfactor: list[np.ndarray],
    chains_idx: list[int],
    original_chain_ids: list[str],
    suffix: str,
) -> None:
    chains_atom_pos_to_pdb(
        str(output_path),
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_type=chains_res_type,
        chains_res_idx=chains_res_idx,
        chains_idx=chains_idx,
        chains_bfactor=chains_bfactor,
        suffix=suffix,
    )

    if suffix.lower() == "cif":
        chain_id_map = {}
        for writer_idx, original_chain_id in zip(chains_idx, original_chain_ids):
            original_chain_id = str(original_chain_id).strip()
            if not original_chain_id:
                continue
            writer_chain_id = chain_names[writer_idx]
            if writer_chain_id != original_chain_id:
                chain_id_map[writer_chain_id] = original_chain_id
        _rewrite_chain_ids_in_cif(output_path, chain_id_map)


def split_structure_protein_na(
    structure_path: Path,
    output_dir: Path,
    *,
    prefix: str,
    suffix: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_chain_ids = _collect_original_chain_ids(structure_path)

    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
        str(structure_path),
        keep_valid=False,
        return_bfactor=True,
    )
    if len(atom_pos) == 0:
        return []

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

    protein_atom_pos = []
    protein_atom_mask = []
    protein_res_type = []
    protein_res_idx = []
    protein_bfactor = []
    protein_chain_idx = []
    protein_chain_ids = []

    na_atom_pos = []
    na_atom_mask = []
    na_res_type = []
    na_res_idx = []
    na_bfactor = []
    na_chain_idx = []
    na_chain_ids = []

    n_chain = len(chains_atom_pos)
    for chain_local_idx in range(n_chain):
        original_chain_id = original_chain_ids[chain_local_idx] if chain_local_idx < len(original_chain_ids) else ""
        writer_chain_index = _resolve_writer_chain_index(original_chain_id, chain_local_idx)

        chain_atom_pos = np.asarray(chains_atom_pos[chain_local_idx])
        chain_atom_mask = np.asarray(chains_atom_mask[chain_local_idx])
        chain_res_type = np.asarray(chains_res_type[chain_local_idx])
        chain_res_idx = np.asarray(chains_res_idx[chain_local_idx])
        chain_bfactor = np.asarray(chains_bfactor[chain_local_idx])

        prot_mask = chain_res_type < 20
        na_mask = (chain_res_type >= 20) & (chain_res_type < 28)

        if np.any(prot_mask):
            protein_atom_pos.append(chain_atom_pos[prot_mask])
            protein_atom_mask.append(chain_atom_mask[prot_mask])
            protein_res_type.append(chain_res_type[prot_mask])
            protein_res_idx.append(chain_res_idx[prot_mask])
            protein_bfactor.append(chain_bfactor[prot_mask])
            protein_chain_idx.append(writer_chain_index)
            protein_chain_ids.append(original_chain_id)

        if np.any(na_mask):
            na_atom_pos.append(chain_atom_pos[na_mask])
            na_atom_mask.append(chain_atom_mask[na_mask])
            na_res_type.append(chain_res_type[na_mask])
            na_res_idx.append(chain_res_idx[na_mask])
            na_bfactor.append(chain_bfactor[na_mask])
            na_chain_idx.append(writer_chain_index)
            na_chain_ids.append(original_chain_id)

    written_paths = []

    if protein_atom_pos:
        prot_output = output_dir / f"{prefix}_prot.{suffix}"
        _write_subset(
            prot_output,
            chains_atom_pos=protein_atom_pos,
            chains_atom_mask=protein_atom_mask,
            chains_res_type=protein_res_type,
            chains_res_idx=protein_res_idx,
            chains_bfactor=protein_bfactor,
            chains_idx=protein_chain_idx,
            original_chain_ids=protein_chain_ids,
            suffix=suffix,
        )
        n_res = int(sum(len(x) for x in protein_atom_pos))
        print(f"write\tkind=prot\toutput={prot_output}\tchains={len(protein_atom_pos)}\tresidues={n_res}")
        written_paths.append(prot_output)

    if na_atom_pos:
        na_output = output_dir / f"{prefix}_na.{suffix}"
        _write_subset(
            na_output,
            chains_atom_pos=na_atom_pos,
            chains_atom_mask=na_atom_mask,
            chains_res_type=na_res_type,
            chains_res_idx=na_res_idx,
            chains_bfactor=na_bfactor,
            chains_idx=na_chain_idx,
            original_chain_ids=na_chain_ids,
            suffix=suffix,
        )
        n_res = int(sum(len(x) for x in na_atom_pos))
        print(f"write\tkind=na\toutput={na_output}\tchains={len(na_atom_pos)}\tresidues={n_res}")
        written_paths.append(na_output)

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split one structure file into combined protein and nucleic-acid files.",
    )
    parser.add_argument("structure", help="Input PDB/mmCIF file")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output file prefix. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--suffix",
        choices=("cif", "pdb"),
        default="cif",
        help="Output structure format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    structure_path = Path(args.structure).expanduser().resolve()
    if not structure_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else structure_path.parent
    )
    prefix = (args.prefix or structure_path.stem).strip()
    if not prefix:
        raise SystemExit("Output prefix must not be empty")

    written_paths = split_structure_protein_na(
        structure_path,
        output_dir,
        prefix=prefix,
        suffix=args.suffix,
    )
    if not written_paths:
        raise SystemExit(f"No supported protein/NA residues were written for {structure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
