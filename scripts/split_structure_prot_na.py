#!/usr/bin/env python3
"""Split one structure file into protein-only and nucleic-acid-only CIF files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser, Select


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


def _build_parser_for_path(path: Path):
    suffix = path.suffix.lower()
    if suffix in PDB_SUFFIXES:
        return PDBParser(QUIET=True)
    if suffix in CIF_SUFFIXES:
        return MMCIFParser(QUIET=True)
    raise ValueError(f"Unsupported structure format: {path}")


def _normalize_resname(residue) -> str:
    return residue.get_resname().strip().upper()


def _is_standard_polymer_residue(residue) -> bool:
    try:
        hetfield, _resseq, _icode = residue.get_id()
    except Exception:
        return False
    return hetfield == " "


def _is_protein_residue(residue) -> bool:
    if not _is_standard_polymer_residue(residue):
        return False
    resname = _normalize_resname(residue)
    return rc.restype3_is_prot(resname)


def _is_na_residue(residue) -> bool:
    if not _is_standard_polymer_residue(residue):
        return False
    resname = _normalize_resname(residue)
    return resname in rc.index_to_nuc


class _ResidueKindSelect(Select):
    def __init__(self, kind: str):
        if kind not in {"prot", "na"}:
            raise ValueError(f"Unsupported kind: {kind}")
        self.kind = kind

    def accept_model(self, model):
        return 1 if int(model.id) == 0 else 0

    def accept_chain(self, chain):
        for residue in chain:
            if self.accept_residue(residue):
                return 1
        return 0

    def accept_residue(self, residue):
        if self.kind == "prot":
            return 1 if _is_protein_residue(residue) else 0
        return 1 if _is_na_residue(residue) else 0

    def accept_atom(self, atom):
        return 1


def _count_kind_residues(structure, kind: str) -> int:
    count = 0
    model = structure[0]
    for chain in model:
        for residue in chain:
            if kind == "prot":
                count += int(_is_protein_residue(residue))
            else:
                count += int(_is_na_residue(residue))
    return count


def _default_prefix(structure_path: Path) -> str:
    return structure_path.stem


def _write_kind(structure, output_path: Path, kind: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(output_path), select=_ResidueKindSelect(kind))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split one structure into <prefix>_prot.cif and <prefix>_na.cif.",
    )
    parser.add_argument("structure", help="Input structure file (.pdb/.ent/.cif/.mmcif)")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory. Default: input file directory.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output prefix. Default: input file stem.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
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
    prefix = (args.prefix or _default_prefix(structure_path)).strip()
    if not prefix:
        raise SystemExit("Output prefix must not be empty")

    prot_output = output_dir / f"{prefix}_prot.cif"
    na_output = output_dir / f"{prefix}_na.cif"
    if not args.overwrite:
        existing = [str(path) for path in (prot_output, na_output) if path.exists()]
        if existing:
            raise FileExistsError("Output already exists: " + ", ".join(existing))

    parser = _build_parser_for_path(structure_path)
    structure = parser.get_structure(prefix, str(structure_path))

    prot_count = _count_kind_residues(structure, "prot")
    na_count = _count_kind_residues(structure, "na")

    print(f"# input = {structure_path}")
    print(f"# output_dir = {output_dir}")
    print(f"# prefix = {prefix}")
    print(f"# protein_residues = {prot_count}")
    print(f"# na_residues = {na_count}")

    if prot_count > 0:
        _write_kind(structure, prot_output, "prot")
        print(f"# wrote_prot = {prot_output}")
    else:
        print("# wrote_prot = none")

    if na_count > 0:
        _write_kind(structure, na_output, "na")
        print(f"# wrote_na = {na_output}")
    else:
        print("# wrote_na = none")

    if prot_count == 0 and na_count == 0:
        raise SystemExit("No standard protein or nucleic-acid residues were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
