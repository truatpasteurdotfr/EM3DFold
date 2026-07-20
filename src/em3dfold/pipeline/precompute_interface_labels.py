import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

VALID_RESNAME_3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "A", "G", "C", "U", "DA", "DG", "DC", "DT",
}
PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
NA_RESNAMES = {"A", "G", "C", "U", "DA", "DG", "DC", "DT"}
LABEL_INTERFACE = "interface"
LABEL_SURFACE = "surface"
LABEL_CORE = "core"

# Protein residue maxima follow common Gly-X-Gly reference values used for RSA.
PROTEIN_MAX_SASA = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}
# Approximate nucleic-acid maxima used for residue-wise SASA normalization.
NA_MAX_SASA = {
    "A": 400.0,
    "G": 410.0,
    "C": 350.0,
    "U": 340.0,
    "DA": 400.0,
    "DG": 410.0,
    "DC": 350.0,
    "DT": 340.0,
}
MAX_SASA = {**PROTEIN_MAX_SASA, **NA_MAX_SASA}


def add_args(parser):
    parser.add_argument("--input", "-i", required=True, help="Input protein-NA complex structure (.pdb/.cif/.mmcif)")
    parser.add_argument("--output", "-o", required=True, help="Output TSV file with one residue label per line")
    parser.add_argument("--interface-cutoff", type=float, default=6.0, help="Heavy-atom cutoff in Angstrom for protein-NA interface")
    parser.add_argument("--surface-rsa-threshold", type=float, default=0.20, help="Residue RSA threshold for surface if not interface")
    return parser


def _build_parser_for_path(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return PDBParser(QUIET=True)
    if suffix in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    raise ValueError(f"Unsupported structure format: {path}")


def _get_mol_type(resname: str) -> str:
    if resname in PROTEIN_RESNAMES:
        return "protein"
    if resname in NA_RESNAMES:
        return "nucleic"
    raise ValueError(f"Unsupported residue name: {resname}")


def _get_rsa(resname: str, sasa: float) -> float:
    max_sasa = float(MAX_SASA[resname])
    if max_sasa <= 0.0:
        return 0.0
    return float(sasa) / max_sasa


def _is_heavy_atom(atom) -> bool:
    element = (getattr(atom, "element", "") or "").strip().upper()
    if element == "H":
        return False
    atom_name = atom.get_name().strip().upper()
    return not atom_name.startswith("H")


def _extract_polymer_residues(structure_path: str):
    path = Path(structure_path)
    parser = _build_parser_for_path(path)
    structure = parser.get_structure(path.stem or "model", str(path))
    model = structure[0]

    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    residues = []
    for chain in model:
        prev_residue_number = None
        for residue in chain:
            residue_number = residue.get_id()[1]
            if prev_residue_number is None or residue_number != prev_residue_number:
                resname_3 = residue.get_resname().strip()
                if resname_3 not in VALID_RESNAME_3:
                    continue
                prev_residue_number = residue_number
            hetfield, resseq, icode = residue.get_id()
            if hetfield != " ":
                continue

            atom_coords = []
            for atom in residue:
                if _is_heavy_atom(atom):
                    atom_coords.append(np.asarray(atom.get_coord(), dtype=np.float32))
            if not atom_coords:
                continue

            sasa = float(getattr(residue, "sasa", 0.0))
            residues.append(
                {
                    "key": (str(chain.id).strip(), str(resseq), str(icode).strip(), resname_3),
                    "chain": str(chain.id).strip(),
                    "resid": str(resseq),
                    "icode": str(icode).strip(),
                    "resname": resname_3,
                    "mol": _get_mol_type(resname_3),
                    "coords": np.asarray(atom_coords, dtype=np.float32),
                    "sasa": sasa,
                    "rsa": _get_rsa(resname_3, sasa),
                }
            )
    return residues


def _assign_interface_flags(residues, cutoff: float):
    prot_atoms = []
    na_atoms = []
    for residue in residues:
        if residue["mol"] == "protein":
            prot_atoms.append(residue["coords"])
        else:
            na_atoms.append(residue["coords"])

    if prot_atoms:
        prot_tree = cKDTree(np.concatenate(prot_atoms, axis=0))
    else:
        prot_tree = None

    if na_atoms:
        na_tree = cKDTree(np.concatenate(na_atoms, axis=0))
    else:
        na_tree = None

    interface_flags = np.zeros((len(residues),), dtype=bool)
    min_cross_dist = np.full((len(residues),), np.inf, dtype=np.float32)

    for idx, residue in enumerate(residues):
        own_coords = residue["coords"]
        if residue["mol"] == "protein":
            if na_tree is None:
                continue
            dists, _ = na_tree.query(own_coords, k=1, distance_upper_bound=np.inf)
            finite = np.isfinite(dists)
            if np.any(finite):
                min_cross_dist[idx] = float(np.min(dists[finite]))
            neighbor_lists = na_tree.query_ball_point(own_coords, r=cutoff)
        else:
            if prot_tree is None:
                continue
            dists, _ = prot_tree.query(own_coords, k=1, distance_upper_bound=np.inf)
            finite = np.isfinite(dists)
            if np.any(finite):
                min_cross_dist[idx] = float(np.min(dists[finite]))
            neighbor_lists = prot_tree.query_ball_point(own_coords, r=cutoff)
        if any(len(lst) > 0 for lst in neighbor_lists):
            interface_flags[idx] = True
    return interface_flags, min_cross_dist


def _assign_label(is_interface: bool, rsa: float, surface_rsa_threshold: float) -> str:
    if is_interface:
        return LABEL_INTERFACE
    if float(rsa) > float(surface_rsa_threshold):
        return LABEL_SURFACE
    return LABEL_CORE


def main(args):
    residues = _extract_polymer_residues(args.input)
    if not residues:
        raise ValueError(f"No supported polymer residues found in {args.input}")

    interface_flags, min_cross_dist = _assign_interface_flags(residues, float(args.interface_cutoff))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(["chain", "resid", "icode", "resname", "mol", "label", "sasa", "rsa", "min_cross_dist"])
        for idx, residue in enumerate(residues):
            label = _assign_label(interface_flags[idx], residue["rsa"], float(args.surface_rsa_threshold))
            writer.writerow([
                residue["chain"],
                residue["resid"],
                residue["icode"],
                residue["resname"],
                residue["mol"],
                label,
                f"{float(residue['sasa']):.6f}",
                f"{float(residue['rsa']):.6f}",
                "inf" if not np.isfinite(min_cross_dist[idx]) else f"{float(min_cross_dist[idx]):.6f}",
            ])

    label_counts = {LABEL_INTERFACE: 0, LABEL_SURFACE: 0, LABEL_CORE: 0}
    for idx, residue in enumerate(residues):
        label = _assign_label(interface_flags[idx], residue["rsa"], float(args.surface_rsa_threshold))
        label_counts[label] += 1

    print(f"# input = {Path(args.input).resolve()}")
    print(f"# output = {output_path}")
    print(f"# interface_cutoff = {float(args.interface_cutoff):.4f}")
    print(f"# surface_rsa_threshold = {float(args.surface_rsa_threshold):.4f}")
    print(f"# residues_total = {len(residues)}")
    print(f"# interface = {label_counts[LABEL_INTERFACE]}")
    print(f"# surface = {label_counts[LABEL_SURFACE]}")
    print(f"# core = {label_counts[LABEL_CORE]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    main(parser.parse_args())
