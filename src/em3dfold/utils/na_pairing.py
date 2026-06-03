import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from em3dfold.polymer_utils import residue_constants as rc


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
    source_index: int
    chain_id: int
    chain_order_index: int
    residue_index: int
    resname: str
    sugar_anchor: np.ndarray
    base_centroid: np.ndarray
    base_normal: np.ndarray
    base_atoms: np.ndarray


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


def _collect_base_atoms_from_atom23(
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    atom_names: Sequence[str],
) -> np.ndarray:
    base_atoms: List[np.ndarray] = []
    for atom_name, present, atom_pos in zip(atom_names, atom_mask, atom_positions):
        if not bool(present):
            continue
        atom_name = atom_name.strip()
        if len(atom_name) == 0:
            continue
        if atom_name.startswith("H"):
            continue
        if atom_name in SUGAR_AND_PHOSPHATE_ATOMS:
            continue
        base_atoms.append(np.asarray(atom_pos, dtype=np.float32))
    if not base_atoms:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(base_atoms, axis=0)


def _build_nucleotide_records_from_atom23(
    atomc_positions: np.ndarray,
    atomc_mask: np.ndarray,
    aatype: np.ndarray,
    chain_index: np.ndarray,
    residue_index: np.ndarray,
) -> List[NucleotideRecord]:
    atomc_positions = np.asarray(atomc_positions, dtype=np.float32)
    atomc_mask = np.asarray(atomc_mask, dtype=bool)
    aatype = np.asarray(aatype, dtype=np.int32)
    chain_index = np.asarray(chain_index, dtype=np.int32)
    residue_index = np.asarray(residue_index, dtype=np.int32)

    chain_counts: Dict[int, int] = {}
    records: List[NucleotideRecord] = []
    for source_index in range(len(aatype)):
        residue_type_index = int(aatype[source_index])
        if residue_type_index < 0 or residue_type_index >= len(rc.index_to_restype_3):
            continue
        resname = rc.index_to_restype_3[residue_type_index]
        if resname not in NA_RESNAMES:
            continue

        atom_names = rc.restype_name_to_atomc_names[resname]
        atoms_by_name = {
            atom_name.strip(): np.asarray(atom_pos, dtype=np.float32)
            for atom_name, atom_present, atom_pos in zip(
                atom_names,
                atomc_mask[source_index],
                atomc_positions[source_index],
            )
            if bool(atom_present) and len(atom_name.strip()) > 0
        }
        sugar_anchor = atoms_by_name.get("C1'")
        if sugar_anchor is None:
            sugar_anchor = atoms_by_name.get("C4'")
        base_atoms = _collect_base_atoms_from_atom23(
            atom_positions=atomc_positions[source_index],
            atom_mask=atomc_mask[source_index],
            atom_names=atom_names,
        )
        if sugar_anchor is None or len(base_atoms) < 3:
            continue

        chain_id = int(chain_index[source_index])
        chain_order_index = int(chain_counts.get(chain_id, 0))
        chain_counts[chain_id] = chain_order_index + 1
        base_centroid = np.mean(base_atoms, axis=0).astype(np.float32)
        base_normal = _fit_plane_normal(base_atoms)
        records.append(
            NucleotideRecord(
                source_index=source_index,
                chain_id=chain_id,
                chain_order_index=chain_order_index,
                residue_index=int(residue_index[source_index]),
                resname=resname,
                sugar_anchor=np.asarray(sugar_anchor, dtype=np.float32),
                base_centroid=base_centroid,
                base_normal=base_normal,
                base_atoms=base_atoms,
            )
        )
    return records


def build_pair_score_matrix(
    records: Sequence[NucleotideRecord],
    same_chain_min_sep: int = 3,
    centroid_max: float = 9.5,
    vertical_max: float = 3.8,
    lateral_max: float = 5.5,
    sugar_max: float = 16.0,
    min_base_max: float = 4.5,
    min_cosine: float = 0.82,
) -> np.ndarray:
    num_res = len(records)
    score_matrix = np.zeros((num_res, num_res), dtype=np.float32)

    for i in range(num_res):
        for j in range(i + 1, num_res):
            left = records[i]
            right = records[j]
            if left.chain_id == right.chain_id:
                residue_gap = abs(int(left.residue_index) - int(right.residue_index))
                if residue_gap <= int(same_chain_min_sep):
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

    return score_matrix


def build_pair_score_matrix_from_atom23(
    atomc_positions: np.ndarray,
    atomc_mask: np.ndarray,
    aatype: np.ndarray,
    chain_index: np.ndarray,
    residue_index: np.ndarray,
    same_chain_min_sep: int = 3,
    centroid_max: float = 9.5,
    vertical_max: float = 3.8,
    lateral_max: float = 5.5,
    sugar_max: float = 16.0,
    min_base_max: float = 4.5,
    min_cosine: float = 0.82,
) -> Tuple[np.ndarray, List[NucleotideRecord]]:
    records = _build_nucleotide_records_from_atom23(
        atomc_positions=atomc_positions,
        atomc_mask=atomc_mask,
        aatype=aatype,
        chain_index=chain_index,
        residue_index=residue_index,
    )
    compact_score_matrix = build_pair_score_matrix(
        records=records,
        same_chain_min_sep=same_chain_min_sep,
        centroid_max=centroid_max,
        vertical_max=vertical_max,
        lateral_max=lateral_max,
        sugar_max=sugar_max,
        min_base_max=min_base_max,
        min_cosine=min_cosine,
    )

    full_score_matrix = np.zeros((len(aatype), len(aatype)), dtype=np.float32)
    if len(records) == 0:
        return full_score_matrix, records

    source_indices = np.asarray([record.source_index for record in records], dtype=np.int32)
    full_score_matrix[np.ix_(source_indices, source_indices)] = compact_score_matrix
    return full_score_matrix, records


def build_pairing_score_matrix_from_atom23(
    atomc_positions: np.ndarray,
    atomc_mask: np.ndarray,
    aatype: np.ndarray,
    chain_index: np.ndarray,
    residue_index: np.ndarray,
    same_chain_min_sep: int = 3,
    centroid_max: float = 9.5,
    vertical_max: float = 3.8,
    lateral_max: float = 5.5,
    sugar_max: float = 16.0,
    min_base_max: float = 4.5,
    min_cosine: float = 0.82,
) -> Tuple[np.ndarray, List[NucleotideRecord]]:
    return build_pair_score_matrix_from_atom23(
        atomc_positions=atomc_positions,
        atomc_mask=atomc_mask,
        aatype=aatype,
        chain_index=chain_index,
        residue_index=residue_index,
        same_chain_min_sep=same_chain_min_sep,
        centroid_max=centroid_max,
        vertical_max=vertical_max,
        lateral_max=lateral_max,
        sugar_max=sugar_max,
        min_base_max=min_base_max,
        min_cosine=min_cosine,
    )


def threshold_pair_score_matrix(score_matrix: np.ndarray, score_threshold: float = 0.45) -> np.ndarray:
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    pair_label = (score_matrix >= float(score_threshold)).astype(np.int32)
    np.fill_diagonal(pair_label, 0)
    return pair_label
