import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Bio.PDB import MMCIFParser, PDBParser

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from em3dfold.utils.sec_struct_utils import pair_matrix_to_dot_bracket, write_dbn


RNA_RESNAMES = {"A", "C", "G", "U", "I"}
DNA_RESNAMES = {"DA", "DC", "DG", "DT", "DI", "DU"}
NA_RESNAMES = RNA_RESNAMES | DNA_RESNAMES
SUGAR_AND_PHOSPHATE_ATOMS = {
    "P",
    "OP1",
    "OP2",
    "OP3",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
}


@dataclass
class NucleotideRecord:
    global_index: int
    chain_id: str
    chain_order_index: int
    resseq: int
    icode: str
    resname: str
    sugar_anchor: np.ndarray
    base_centroid: np.ndarray
    base_normal: np.ndarray
    base_atoms: np.ndarray


def _resname_to_dbn_token(resname: str) -> str:
    normalized = resname.strip().upper()
    if normalized in {"A", "DA"}:
        return "A"
    if normalized in {"C", "DC"}:
        return "C"
    if normalized in {"G", "DG"}:
        return "G"
    if normalized in {"U", "DT", "DU"}:
        return "U"
    return "N"


def _log_info(message: str):
    print(f"[Info] {message}", flush=True)


def _load_structure(structure_path: str):
    suffix = Path(structure_path).suffix.lower()
    if suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    elif suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure format: {structure_path}")
    return parser.get_structure("structure", structure_path)


def _is_nucleotide_residue(residue) -> bool:
    return residue.resname.strip().upper() in NA_RESNAMES


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return vector / norm


def _fit_plane_normal(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    return _normalize_vector(normal)


def _collect_base_atoms(residue) -> np.ndarray:
    base_atoms: List[np.ndarray] = []
    for atom in residue:
        atom_name = atom.name.strip()
        if atom_name.startswith("H"):
            continue
        if atom_name in SUGAR_AND_PHOSPHATE_ATOMS:
            continue
        base_atoms.append(np.asarray(atom.coord, dtype=np.float32))
    if not base_atoms:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(base_atoms, axis=0)


def _build_nucleotide_records(structure_path: str, selected_chains: Optional[Sequence[str]] = None):
    structure = _load_structure(structure_path)
    model = next(structure.get_models())
    chain_filter = None if selected_chains is None else {chain_id.strip() for chain_id in selected_chains}

    records: List[NucleotideRecord] = []
    for chain in model:
        if chain_filter is not None and chain.id not in chain_filter:
            continue

        chain_order_index = 0
        for residue in chain:
            if not _is_nucleotide_residue(residue):
                continue

            atoms = {atom.name.strip(): np.asarray(atom.coord, dtype=np.float32) for atom in residue}
            sugar_anchor = atoms.get("C1'")
            if sugar_anchor is None:
                sugar_anchor = atoms.get("C4'")
            base_atoms = _collect_base_atoms(residue)
            if sugar_anchor is None or len(base_atoms) < 3:
                continue

            base_centroid = np.mean(base_atoms, axis=0).astype(np.float32)
            base_normal = _fit_plane_normal(base_atoms)
            hetfield, resseq, icode = residue.get_id()
            if hetfield != " ":
                continue

            records.append(
                NucleotideRecord(
                    global_index=len(records),
                    chain_id=chain.id,
                    chain_order_index=chain_order_index,
                    resseq=int(resseq),
                    icode=str(icode).strip(),
                    resname=residue.resname.strip().upper(),
                    sugar_anchor=np.asarray(sugar_anchor, dtype=np.float32),
                    base_centroid=base_centroid,
                    base_normal=base_normal,
                    base_atoms=base_atoms,
                )
            )
            chain_order_index += 1
    return records


def _gaussian_score(value: float, center: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    z = (value - center) / sigma
    return float(math.exp(-(z * z)))


def _parallel_score(cos_parallel: float, min_cosine: float) -> float:
    if cos_parallel <= min_cosine:
        return 0.0
    return float((cos_parallel - min_cosine) / max(1e-6, 1.0 - min_cosine))


def _pairwise_min_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    diffs = points_a[:, None, :] - points_b[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return float(np.min(dists))


def _compute_pair_features(left: NucleotideRecord, right: NucleotideRecord) -> Dict[str, float]:
    delta = right.base_centroid - left.base_centroid
    d_centroid = float(np.linalg.norm(delta))
    cos_parallel = float(np.clip(abs(np.dot(left.base_normal, right.base_normal)), 0.0, 1.0))
    d_vert_left = abs(float(np.dot(delta, left.base_normal)))
    d_vert_right = abs(float(np.dot(delta, right.base_normal)))
    d_vertical = 0.5 * (d_vert_left + d_vert_right)
    d_lateral_sq = max(0.0, d_centroid * d_centroid - d_vertical * d_vertical)
    d_lateral = float(math.sqrt(d_lateral_sq))
    d_sugar = float(np.linalg.norm(right.sugar_anchor - left.sugar_anchor))
    d_min_base = _pairwise_min_distance(left.base_atoms, right.base_atoms)
    return {
        "d_centroid": d_centroid,
        "cos_parallel": cos_parallel,
        "d_vertical": d_vertical,
        "d_lateral": d_lateral,
        "d_sugar": d_sugar,
        "d_min_base": d_min_base,
    }


def _compute_pair_score(
    features: Dict[str, float],
    centroid_max: float,
    vertical_max: float,
    lateral_max: float,
    sugar_max: float,
    min_base_max: float,
    min_cosine: float,
) -> float:
    if features["d_centroid"] > centroid_max:
        return 0.0
    if features["d_vertical"] > vertical_max:
        return 0.0
    if features["d_lateral"] > lateral_max:
        return 0.0
    if features["d_sugar"] > sugar_max:
        return 0.0
    if features["d_min_base"] > min_base_max:
        return 0.0
    if features["cos_parallel"] < min_cosine:
        return 0.0

    component_scores = [
        _gaussian_score(features["d_centroid"], center=5.8, sigma=1.8),
        _parallel_score(features["cos_parallel"], min_cosine=min_cosine),
        _gaussian_score(features["d_vertical"], center=2.8, sigma=0.9),
        _gaussian_score(features["d_lateral"], center=0.8, sigma=2.4),
        _gaussian_score(features["d_sugar"], center=10.5, sigma=3.2),
        _gaussian_score(features["d_min_base"], center=2.9, sigma=1.1),
    ]
    return float(np.clip(np.mean(component_scores), 0.0, 1.0))


def build_pair_matrix(
    records: Sequence[NucleotideRecord],
    same_chain_min_sep: int = 3,
    centroid_max: float = 9.5,
    vertical_max: float = 3.8,
    lateral_max: float = 5.5,
    sugar_max: float = 16.0,
    min_base_max: float = 4.5,
    min_cosine: float = 0.82,
    score_threshold: float = 0.45,
    top_k: int = 3,
):
    num_res = len(records)
    score_matrix = np.zeros((num_res, num_res), dtype=np.float32)
    feature_rows: List[Dict[str, object]] = []

    for i in range(num_res):
        for j in range(i + 1, num_res):
            left = records[i]
            right = records[j]
            if (
                left.chain_id == right.chain_id
                and abs(left.chain_order_index - right.chain_order_index) <= same_chain_min_sep
            ):
                continue

            features = _compute_pair_features(left, right)
            score = _compute_pair_score(
                features,
                centroid_max=centroid_max,
                vertical_max=vertical_max,
                lateral_max=lateral_max,
                sugar_max=sugar_max,
                min_base_max=min_base_max,
                min_cosine=min_cosine,
            )
            if score <= 0.0:
                continue

            score_matrix[i, j] = score
            score_matrix[j, i] = score
            feature_rows.append(
                {
                    "left_index": i,
                    "right_index": j,
                    "score": score,
                    **features,
                }
            )

    pair_mask = score_matrix >= float(score_threshold)
    top_pairs = []
    for i in range(num_res):
        if top_k <= 0:
            continue
        row = score_matrix[i].copy()
        row[i] = 0.0
        candidate_indices = np.argsort(row)[::-1]
        kept = []
        for idx in candidate_indices:
            if row[idx] <= 0.0:
                break
            kept.append({"partner_index": int(idx), "score": float(row[idx])})
            if len(kept) >= top_k:
                break
        top_pairs.append({"index": i, "partners": kept})

    feature_rows = sorted(feature_rows, key=lambda item: item["score"], reverse=True)
    return score_matrix, pair_mask.astype(np.uint8), feature_rows, top_pairs


def _default_output_prefix(structure_path: str) -> str:
    input_path = Path(structure_path)
    if input_path.suffix.lower() in {".pdb", ".cif"}:
        input_path = input_path.with_suffix("")
    return str(input_path) + ".pairmat"


def get_args():
    parser = argparse.ArgumentParser(
        description="Build a relaxed nucleotide pairing score matrix from a structure."
    )
    parser.add_argument("-i", "--input", required=True, help="Input PDB/mmCIF structure")
    parser.add_argument(
        "-c",
        "--chains",
        default=None,
        help="Optional comma-separated chain IDs to keep",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default=None,
        help="Output prefix. Defaults to <input>.pairmat",
    )
    parser.add_argument("--same-chain-min-sep", type=int, default=3)
    parser.add_argument("--centroid-max", type=float, default=9.5)
    parser.add_argument("--vertical-max", type=float, default=3.8)
    parser.add_argument("--lateral-max", type=float, default=5.5)
    parser.add_argument("--sugar-max", type=float, default=16.0)
    parser.add_argument("--min-base-max", type=float, default=4.5)
    parser.add_argument("--min-cosine", type=float, default=0.82)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = get_args()
    output_prefix = args.output_prefix or _default_output_prefix(args.input)
    chain_ids = None
    if args.chains is not None and args.chains.strip():
        chain_ids = [chain_id.strip() for chain_id in args.chains.split(",") if chain_id.strip()]

    _log_info("load structure and collect nucleotide residues")
    records = _build_nucleotide_records(args.input, selected_chains=chain_ids)
    if len(records) == 0:
        raise ValueError("No nucleotide residues with enough base atoms were found")

    _log_info(f"build relaxed pairing matrix for {len(records)} nucleotide residues")
    score_matrix, pair_mask, pair_rows, top_pairs = build_pair_matrix(
        records,
        same_chain_min_sep=args.same_chain_min_sep,
        centroid_max=args.centroid_max,
        vertical_max=args.vertical_max,
        lateral_max=args.lateral_max,
        sugar_max=args.sugar_max,
        min_base_max=args.min_base_max,
        min_cosine=args.min_cosine,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
    )

    score_path = output_prefix + ".score.npy"
    mask_path = output_prefix + ".mask.npy"
    dbn_path = output_prefix + ".dbn"
    residues_path = output_prefix + ".residues.json"
    summary_path = output_prefix + ".summary.json"
    sequence = "".join(_resname_to_dbn_token(record.resname) for record in records)
    dot_bracket, dbn_pairs = pair_matrix_to_dot_bracket(
        score_matrix,
        threshold=args.score_threshold,
        enforce_one_partner=True,
        allow_pseudoknots=True,
        min_separation=args.same_chain_min_sep,
    )

    residue_rows = [
        {
            "index": record.global_index,
            "chain_id": record.chain_id,
            "chain_order_index": record.chain_order_index,
            "resseq": record.resseq,
            "icode": record.icode,
            "resname": record.resname,
        }
        for record in records
    ]
    summary = {
        "input": args.input,
        "num_nucleotide_residues": len(records),
        "score_threshold": args.score_threshold,
        "same_chain_min_sep": args.same_chain_min_sep,
        "thresholds": {
            "centroid_max": args.centroid_max,
            "vertical_max": args.vertical_max,
            "lateral_max": args.lateral_max,
            "sugar_max": args.sugar_max,
            "min_base_max": args.min_base_max,
            "min_cosine": args.min_cosine,
        },
        "num_thresholded_pairs": int(np.sum(pair_mask) // 2),
        "dbn_path": dbn_path,
        "dbn_sequence": sequence,
        "dbn": dot_bracket,
        "dbn_pairs": [[int(i), int(j)] for i, j in dbn_pairs],
        "top_pairs": top_pairs,
        "best_pairs": pair_rows[: min(200, len(pair_rows))],
    }

    np.save(score_path, score_matrix)
    np.save(mask_path, pair_mask)
    write_dbn(dbn_path, dot_bracket=dot_bracket, sequence=sequence, name=Path(output_prefix).name)
    Path(residues_path).write_text(json.dumps(residue_rows, indent=2), encoding="utf-8")
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _log_info(f"write pair score matrix to {score_path}")
    _log_info(f"write pair mask matrix to {mask_path}")
    _log_info(f"write dbn to {dbn_path}")
    _log_info(f"write residue metadata to {residues_path}")
    _log_info(f"write summary to {summary_path}")
    print(sequence, flush=True)
    print(dot_bracket, flush=True)


if __name__ == "__main__":
    main()
