#!/usr/bin/env python3
"""Merge multiple mmCIF files into one mmCIF file."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from Bio.PDB import MMCIFParser


def _ensure_repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_ensure_repo_src_on_path()
from em3dfold.io.pdbio import CIFXIO, chain_names, fix_quotes  # noqa: E402


CIF_SUFFIXES = {".cif", ".mmcif"}


def _is_cif_path(path: Path) -> bool:
    return path.suffix.lower() in CIF_SUFFIXES


def _next_available_chain_id(used: set[str]) -> str:
    for chain_id in chain_names:
        if chain_id not in used:
            return chain_id
    raise ValueError("ran out of available chain ids while merging CIF files")


def merge_cif_files(input_paths: list[Path], output_path: Path) -> tuple[list[tuple[str, str, str]], list[str]]:
    parser = MMCIFParser(QUIET=True)
    if not input_paths:
        raise ValueError("no input CIF files provided")

    base_structure = None
    base_model = None
    used_chain_ids: set[str] = set()
    rename_records: list[tuple[str, str, str]] = []
    skipped_inputs: list[str] = []

    for input_idx, input_path in enumerate(input_paths):
        if not _is_cif_path(input_path):
            raise ValueError(f"unsupported input format: {input_path}")
        if not input_path.exists():
            skipped_inputs.append(str(input_path))
            continue

        structure = parser.get_structure(input_path.stem or f"input_{input_idx}", str(input_path))
        model = structure[0]

        if base_structure is None:
            base_structure = copy.deepcopy(structure)
            base_model = base_structure[0]
            used_chain_ids = {str(chain.id).strip() for chain in base_model}
            continue

        for chain in model:
            chain_copy = copy.deepcopy(chain)
            original_chain_id = str(chain_copy.id).strip()
            final_chain_id = original_chain_id
            if final_chain_id in used_chain_ids or final_chain_id == "":
                final_chain_id = _next_available_chain_id(used_chain_ids)
                chain_copy.id = final_chain_id
                rename_records.append((str(input_path), original_chain_id or "-", final_chain_id))
            used_chain_ids.add(final_chain_id)
            base_model.add(chain_copy)

    if base_structure is None:
        raise ValueError("failed to load any existing input CIF structure")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = CIFXIO()
    writer.set_structure(base_structure)
    writer.save(str(output_path))
    fix_quotes(str(output_path))
    return rename_records, skipped_inputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multiple CIF/mmCIF files into one output CIF/mmCIF file")
    parser.add_argument("cifs", nargs="+", help="Input CIF/mmCIF files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output CIF/mmCIF file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_paths = [Path(path).expanduser().resolve() for path in args.cifs]
    output_path = Path(args.output).expanduser().resolve()

    rename_records, skipped_inputs = merge_cif_files(input_paths, output_path)
    print(output_path)
    for skipped_input in skipped_inputs:
        print(f"skip\tinput={skipped_input}\treason=missing")
    for input_path, old_chain_id, new_chain_id in rename_records:
        print(f"rename\tinput={input_path}\tchain={old_chain_id}\tnew_chain={new_chain_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
