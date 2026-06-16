#!/usr/bin/env python3
"""Normalize ATOM/HETATM atom-name alignment for CryoRead-style PDB files.

This rewrites only columns 13-16 (the PDB atom-name field) so downstream
fixed-column parsers such as Phenix/MolProbity can recognize standard atom
names like " C1'" instead of left-aligned variants like "C1' ".
"""

from __future__ import annotations

import argparse
from pathlib import Path


TWO_CHAR_ELEMENTS = {
    "BR",
    "CL",
    "NA",
    "MG",
    "ZN",
    "FE",
    "CA",
    "MN",
    "CU",
    "CO",
    "NI",
    "CD",
    "HG",
    "SE",
}


def _infer_element(atom_name: str, raw_element: str) -> str:
    element = raw_element.strip().upper()
    if element:
        return element

    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return ""

    if len(letters) >= 2 and letters[:2].upper() in TWO_CHAR_ELEMENTS:
        return letters[:2].upper()
    return letters[0].upper()


def format_atom_name(raw_atom_name: str, raw_element: str) -> str:
    atom_name = raw_atom_name.strip()
    if not atom_name:
        return raw_atom_name

    if len(atom_name) >= 4:
        return atom_name[:4]

    element = _infer_element(atom_name, raw_element)

    # Standard PDB alignment:
    # - names starting with a digit are left-justified
    # - two-character elements are left-justified
    # - otherwise right-justify into columns 13-16
    if atom_name[0].isdigit() or len(element) == 2:
        return atom_name.ljust(4)
    return atom_name.rjust(4)


def fix_pdb_atom_names(input_path: Path, output_path: Path, overwrite: bool = False) -> tuple[int, int]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    total_records = 0
    changed_records = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                total_records += 1
                stripped_newline = line.rstrip("\n")
                padded = stripped_newline.ljust(80)
                old_atom_name = padded[12:16]
                new_atom_name = format_atom_name(old_atom_name, padded[76:78])
                if new_atom_name != old_atom_name:
                    changed_records += 1
                    padded = padded[:12] + new_atom_name + padded[16:]
                fout.write(padded.rstrip() + "\n")
            else:
                fout.write(line)

    return total_records, changed_records


def default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".pdb"
    stem = input_path.with_suffix("")
    return stem.with_name(f"{stem.name}_fixed_atom_names{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fix PDB atom-name alignment in columns 13-16 so strict fixed-column "
            "parsers can recognize standard atom names."
        )
    )
    parser.add_argument("input", type=Path, help="Input PDB file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDB file. Default: <input>_fixed_atom_names.pdb",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input
    output_path = args.output if args.output is not None else default_output_path(input_path)

    total_records, changed_records = fix_pdb_atom_names(
        input_path=input_path,
        output_path=output_path,
        overwrite=bool(args.overwrite),
    )

    print(f"# input  = {input_path}")
    print(f"# output = {output_path}")
    print(f"# atom_records = {total_records}")
    print(f"# changed_atom_name_records = {changed_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
