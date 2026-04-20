#!/usr/bin/env python3
"""Split a PDB/mmCIF structure into per-chain protein-only files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser, Select


def _ensure_repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_ensure_repo_src_on_path()
from em3dfold.polymer_utils import residue_constants as rc  # noqa: E402


PDB_SUFFIXES = {".pdb", ".ent"}
CIF_SUFFIXES = {".cif", ".mmcif"}


def _sanitize_chain_id(chain_id: str, fallback_index: int) -> str:
    chain_id = str(chain_id).strip()
    if not chain_id:
        return f"chain{fallback_index}"
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in chain_id)


def _is_protein_residue(residue) -> bool:
    try:
        hetfield, _resseq, _icode = residue.get_id()
    except Exception:
        return False
    if hetfield != " ":
        return False
    resname = residue.get_resname().strip()
    if resname not in rc.restype_3_to_index:
        return False
    return int(rc.restype_3_to_index[resname]) < 20


class _ChainProteinSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_model(self, model):
        return int(model.id) == 0

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return 1 if _is_protein_residue(residue) else 0

    def accept_atom(self, atom):
        return 1


def _build_parser_for_path(path: Path):
    suffix = path.suffix.lower()
    if suffix in PDB_SUFFIXES:
        return PDBParser(QUIET=True), "pdb"
    if suffix in CIF_SUFFIXES:
        return MMCIFParser(QUIET=True), "cif"
    raise ValueError(f"Unsupported structure format: {path}")


def _build_writer(fmt: str):
    if fmt == "pdb":
        return PDBIO()
    if fmt == "cif":
        return MMCIFIO()
    raise ValueError(f"Unsupported output format: {fmt}")


def split_protein_chains(structure_path: Path, output_dir: Path, *, prefix: str, suffix: str) -> list[Path]:
    parser, _input_fmt = _build_parser_for_path(structure_path)
    structure = parser.get_structure(structure_path.stem or "model", str(structure_path))
    model = structure[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    writer = _build_writer(suffix)
    writer.set_structure(structure)

    for chain_idx, chain in enumerate(model):
        protein_residues = [residue for residue in chain if _is_protein_residue(residue)]
        if not protein_residues:
            print(f"skip\tchain={chain.id}\treason=no_protein_residues")
            continue

        chain_name = _sanitize_chain_id(chain.id, chain_idx)
        output_path = output_dir / f"{prefix}.chain.{chain_name}.{suffix}"
        writer.save(str(output_path), select=_ChainProteinSelect(chain.id))
        print(f"write\tchain={chain.id}\toutput={output_path}")
        written_paths.append(output_path)

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a PDB/mmCIF structure into per-chain protein-only files.",
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

    written_paths = split_protein_chains(
        structure_path,
        output_dir,
        prefix=prefix,
        suffix=args.suffix,
    )
    if not written_paths:
        raise SystemExit(f"No protein chains were written for {structure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
