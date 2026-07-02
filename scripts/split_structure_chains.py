#!/usr/bin/env python3
"""Split a structure into per-chain protein/NA structure files.

This keeps only standard polymer residues from model 0 and writes one or two
files per chain:
- <prefix>.chain.<chain_id>.prot.<suffix>
- <prefix>.chain.<chain_id>.na.<suffix>

The output writer uses em3dfold's internal pdbio utilities instead of the raw
Bio.PDB mmCIF writer, so the generated CIF is compatible with USalign.
"""

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
from em3dfold.io.pdbio import (  # noqa: E402
    chain_names,
    chains_atom_pos_to_pdb,
    convert_to_chains,
    read_pdb,
)


PDB_SUFFIXES = {".pdb", ".ent"}
CIF_SUFFIXES = {".cif", ".mmcif"}


def _sanitize_chain_id(chain_id: str, fallback_index: int) -> str:
    chain_id = str(chain_id).strip()
    if not chain_id:
        return f"chain{fallback_index}"
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in chain_id)


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


def _rewrite_single_chain_cif_asym_id(output_path: Path, chain_id: str) -> None:
    chain_id = str(chain_id).strip()
    if not chain_id:
        return

    lines = output_path.read_text().splitlines()
    header_indices: dict[str, int] = {}
    atom_site_header_order: list[str] = []
    in_atom_site_loop = False
    loop_start = -1
    data_start = -1

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'loop_':
            in_atom_site_loop = False
            atom_site_header_order = []
            header_indices = {}
            loop_start = idx
            data_start = -1
            continue
        if stripped.startswith('_atom_site.'):
            if not in_atom_site_loop:
                in_atom_site_loop = True
            atom_site_header_order.append(stripped)
            header_indices[stripped] = len(atom_site_header_order) - 1
            continue
        if in_atom_site_loop and stripped and not stripped.startswith('_') and not stripped.startswith('#'):
            data_start = idx
            break

    if data_start < 0:
        return

    label_idx = header_indices.get('_atom_site.label_asym_id')
    auth_idx = header_indices.get('_atom_site.auth_asym_id')
    if label_idx is None and auth_idx is None:
        return

    for idx in range(data_start, len(lines)):
        stripped = lines[idx].strip()
        if not stripped or stripped == '#':
            break
        tokens = stripped.split()
        if label_idx is not None and label_idx < len(tokens):
            tokens[label_idx] = chain_id
        if auth_idx is not None and auth_idx < len(tokens):
            tokens[auth_idx] = chain_id
        lines[idx] = ' '.join(tokens)

    output_path.write_text('\n'.join(lines) + '\n')


def _write_chain_subset(
    output_path: Path,
    *,
    atom_pos: np.ndarray,
    atom_mask: np.ndarray,
    res_type: np.ndarray,
    res_idx: np.ndarray,
    bfactor: np.ndarray,
    writer_chain_index: int,
    output_chain_id: str,
    suffix: str,
) -> None:
    chains_atom_pos_to_pdb(
        str(output_path),
        chains_atom_pos=[atom_pos],
        chains_atom_mask=[atom_mask],
        chains_res_type=[res_type],
        chains_res_idx=[res_idx],
        chains_idx=[writer_chain_index],
        chains_bfactor=[bfactor],
        suffix=suffix,
    )
    if suffix.lower() == 'cif':
        _rewrite_single_chain_cif_asym_id(output_path, output_chain_id)


def split_structure_chains(
    structure_path: Path,
    output_dir: Path,
    *,
    prefix: str,
    suffix: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chain_ids = _collect_original_chain_ids(structure_path)

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

    written_paths: list[Path] = []
    n_chain = len(chains_atom_pos)
    for chain_local_idx in range(max(len(chain_ids), n_chain)):
        original_chain_id = chain_ids[chain_local_idx] if chain_local_idx < len(chain_ids) else ""
        chain_name = _sanitize_chain_id(original_chain_id, chain_local_idx)

        if chain_local_idx >= n_chain:
            print(f"skip\tchain={original_chain_id or chain_local_idx}\treason=no_supported_polymer_residues")
            continue

        chain_atom_pos = np.asarray(chains_atom_pos[chain_local_idx])
        chain_atom_mask = np.asarray(chains_atom_mask[chain_local_idx])
        chain_res_type = np.asarray(chains_res_type[chain_local_idx])
        chain_res_idx = np.asarray(chains_res_idx[chain_local_idx])
        chain_bfactor = np.asarray(chains_bfactor[chain_local_idx])

        prot_mask = chain_res_type < 20
        na_mask = (chain_res_type >= 20) & (chain_res_type < 28)
        prot_count = int(np.count_nonzero(prot_mask))
        na_count = int(np.count_nonzero(na_mask))

        if prot_count + na_count <= 0:
            print(f"skip\tchain={original_chain_id or chain_local_idx}\treason=no_supported_polymer_residues")
            continue

        writer_chain_index = _resolve_writer_chain_index(original_chain_id, chain_local_idx)

        if prot_count > 0:
            output_path = output_dir / f"{prefix}.chain.{chain_name}.prot.{suffix}"
            _write_chain_subset(
                output_path,
                atom_pos=chain_atom_pos[prot_mask],
                atom_mask=chain_atom_mask[prot_mask],
                res_type=chain_res_type[prot_mask],
                res_idx=chain_res_idx[prot_mask],
                bfactor=chain_bfactor[prot_mask],
                writer_chain_index=writer_chain_index,
                output_chain_id=original_chain_id or chain_name,
                suffix=suffix,
            )
            print(
                f"write\tchain={original_chain_id or chain_local_idx}\tkind=prot\t"
                f"output={output_path}\tresidues={prot_count}"
            )
            written_paths.append(output_path)

        if na_count > 0:
            output_path = output_dir / f"{prefix}.chain.{chain_name}.na.{suffix}"
            _write_chain_subset(
                output_path,
                atom_pos=chain_atom_pos[na_mask],
                atom_mask=chain_atom_mask[na_mask],
                res_type=chain_res_type[na_mask],
                res_idx=chain_res_idx[na_mask],
                bfactor=chain_bfactor[na_mask],
                writer_chain_index=writer_chain_index,
                output_chain_id=original_chain_id or chain_name,
                suffix=suffix,
            )
            print(
                f"write\tchain={original_chain_id or chain_local_idx}\tkind=na\t"
                f"output={output_path}\tresidues={na_count}"
            )
            written_paths.append(output_path)

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a PDB/mmCIF structure into per-chain protein/NA files.",
    )
    parser.add_argument("structure", help="Input PDB/mmCIF file")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output file prefix. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--suffix",
        choices=("cif", "pdb"),
        default="cif",
        help="Output structure format",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    structure_path = Path(args.structure).expanduser().resolve()
    if not structure_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    prefix = (args.prefix or structure_path.stem).strip()
    if not prefix:
        raise SystemExit("Output prefix must not be empty")

    written_paths = split_structure_chains(
        structure_path,
        output_dir,
        prefix=prefix,
        suffix=args.suffix,
    )
    if not written_paths:
        raise SystemExit(f"No supported polymer chains were written for {structure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
