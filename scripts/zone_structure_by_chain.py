#!/usr/bin/env python3
"""Select residues from structure B that are within a distance cutoff of chains in structure A."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser, Select
from scipy.spatial import KDTree


PDB_SUFFIXES = {".pdb", ".ent"}
CIF_SUFFIXES = {".cif", ".mmcif"}


def _build_parser_for_path(path: Path):
    suffix = path.suffix.lower()
    if suffix in PDB_SUFFIXES:
        return PDBParser(QUIET=True)
    if suffix in CIF_SUFFIXES:
        return MMCIFParser(QUIET=True)
    raise ValueError(f"Unsupported structure format: {path}")


def _build_writer_for_path(path: Path):
    suffix = path.suffix.lower()
    if suffix in PDB_SUFFIXES:
        return PDBIO()
    if suffix in CIF_SUFFIXES:
        return MMCIFIO()
    raise ValueError(f"Unsupported output structure format: {path}")


def _parse_chain_ids(value: str) -> list[str]:
    chain_ids = [item.strip() for item in str(value).split(",") if item.strip()]
    if not chain_ids:
        raise ValueError("At least one non-empty chain id must be provided")
    return chain_ids


def _is_water_residue(residue) -> bool:
    return residue.get_resname().strip().upper() in {"HOH", "WAT", "H2O"}


def _residue_label(chain_id: str, residue) -> str:
    hetfield, resseq, icode = residue.id
    icode = str(icode).strip()
    suffix = icode if icode else "-"
    het = str(hetfield).strip() or "-"
    return f"{chain_id}:{resseq}:{suffix}:{het}:{residue.get_resname().strip()}"


def _collect_chain_atoms(model, chain_ids: list[str]) -> np.ndarray:
    selected_atoms: list[np.ndarray] = []
    missing_chain_ids: list[str] = []
    available_chain_ids = {str(chain.id).strip() for chain in model}

    for chain_id in chain_ids:
        if chain_id not in available_chain_ids:
            missing_chain_ids.append(chain_id)
            continue
        chain = model[chain_id]
        for atom in chain.get_atoms():
            coord = atom.get_coord()
            if coord is not None:
                selected_atoms.append(np.asarray(coord, dtype=np.float32))

    if missing_chain_ids:
        raise ValueError(
            "Missing chain ids in structure A: " + ", ".join(missing_chain_ids)
        )
    if not selected_atoms:
        raise ValueError("No atoms found in the selected chains of structure A")
    return np.asarray(selected_atoms, dtype=np.float32)


def _collect_selected_residues(
    model,
    tree: KDTree,
    distance_cutoff: float,
    *,
    include_water: bool,
) -> set[tuple[str, tuple]]:
    selected: set[tuple[str, tuple]] = set()
    for chain in model:
        chain_id = str(chain.id).strip()
        for residue in chain:
            if not include_water and _is_water_residue(residue):
                continue
            coords = []
            for atom in residue.get_atoms():
                coord = atom.get_coord()
                if coord is not None:
                    coords.append(np.asarray(coord, dtype=np.float32))
            if not coords:
                continue
            distances, _ = tree.query(np.asarray(coords, dtype=np.float32), k=1, workers=-1)
            min_distance = float(np.min(distances))
            if min_distance <= distance_cutoff:
                selected.add((chain_id, residue.id))
    return selected


class _SelectedResidueSelect(Select):
    def __init__(self, selected_residues: set[tuple[str, tuple]], *, include_water: bool):
        self.selected_residues = selected_residues
        self.include_water = include_water

    def accept_model(self, model):
        return 1 if int(model.id) == 0 else 0

    def accept_chain(self, chain):
        chain_id = str(chain.id).strip()
        for residue in chain:
            if not self.include_water and _is_water_residue(residue):
                continue
            if (chain_id, residue.id) in self.selected_residues:
                return 1
        return 0

    def accept_residue(self, residue):
        parent = residue.get_parent()
        chain_id = str(parent.id).strip()
        if not self.include_water and _is_water_residue(residue):
            return 0
        return 1 if (chain_id, residue.id) in self.selected_residues else 0

    def accept_atom(self, atom):
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zone residues from structure B that lie within a distance cutoff of "
            "specified chains in structure A. Both structures must already be in the same frame."
        )
    )
    parser.add_argument("--a", required=True, help="Reference structure A (.pdb/.cif/.mmcif)")
    parser.add_argument(
        "--a-chain",
        required=True,
        help="Comma-separated chain ids from structure A used as the zoning reference, e.g. A or A,B",
    )
    parser.add_argument("--b", required=True, help="Target structure B (.pdb/.cif/.mmcif)")
    parser.add_argument(
        "-d",
        "--distance",
        type=float,
        required=True,
        help="Residue is kept if any atom in that residue is within this distance (Angstrom) of any atom in A chains",
    )
    parser.add_argument("-o", "--output", required=True, help="Output zoned structure path (.pdb/.cif/.mmcif)")
    parser.add_argument(
        "--include-water",
        action="store_true",
        help="Also consider and keep water residues when they are within the cutoff.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    a_path = Path(args.a).expanduser().resolve()
    b_path = Path(args.b).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    chain_ids = _parse_chain_ids(args.a_chain)
    distance_cutoff = float(args.distance)

    if distance_cutoff < 0:
        raise ValueError(f"distance cutoff must be non-negative, got {distance_cutoff}")
    if not a_path.is_file():
        raise FileNotFoundError(f"Structure A not found: {a_path}")
    if not b_path.is_file():
        raise FileNotFoundError(f"Structure B not found: {b_path}")

    a_parser = _build_parser_for_path(a_path)
    b_parser = _build_parser_for_path(b_path)
    a_structure = a_parser.get_structure(a_path.stem or "a", str(a_path))
    b_structure = b_parser.get_structure(b_path.stem or "b", str(b_path))
    a_model = a_structure[0]
    b_model = b_structure[0]

    a_atoms = _collect_chain_atoms(a_model, chain_ids)
    tree = KDTree(a_atoms)
    selected_residues = _collect_selected_residues(
        b_model,
        tree,
        distance_cutoff,
        include_water=bool(args.include_water),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _build_writer_for_path(output_path)
    writer.set_structure(b_structure)
    writer.save(
        str(output_path),
        select=_SelectedResidueSelect(
            selected_residues,
            include_water=bool(args.include_water),
        ),
    )

    per_chain_counts: dict[str, int] = {}
    example_labels: list[str] = []
    for chain_id, residue_id in sorted(selected_residues, key=lambda item: (item[0], item[1][1], str(item[1][2]))):
        per_chain_counts[chain_id] = per_chain_counts.get(chain_id, 0) + 1
        if len(example_labels) < 10:
            residue = b_model[chain_id][residue_id]
            example_labels.append(_residue_label(chain_id, residue))

    print(f"# structure_a = {a_path}")
    print(f"# structure_a_chains = {','.join(chain_ids)}")
    print(f"# structure_b = {b_path}")
    print(f"# distance = {distance_cutoff:.4f}")
    print(f"# include_water = {bool(args.include_water)}")
    print(f"# selected_residues = {len(selected_residues)}")
    print(f"# output = {output_path}")
    for chain_id in sorted(per_chain_counts):
        print(f"chain\t{chain_id}\tresidues={per_chain_counts[chain_id]}")
    if example_labels:
        print("# examples = " + ", ".join(example_labels))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
