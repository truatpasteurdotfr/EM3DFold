from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]


def linear_segment_adjacency(n_segments: int) -> BoolArray:
    adjacency = np.zeros((n_segments, n_segments), dtype=np.bool_)
    for idx in range(max(n_segments - 1, 0)):
        adjacency[idx, idx + 1] = True
        adjacency[idx + 1, idx] = True
    return adjacency


@dataclass(slots=True)
class MRCMap:
    path: Path
    data: FloatArray
    ncrs: tuple[int, int, int]
    ncrsstart: tuple[int, int, int]
    mxyz: tuple[int, int, int]
    cella: FloatArray
    cellb: FloatArray
    mapcrs: tuple[int, int, int]
    origin: FloatArray
    mode: int

    @property
    def voxel_size(self) -> FloatArray:
        return (self.cella / np.asarray(self.mxyz, dtype=np.float32)).astype(np.float32)


@dataclass(slots=True)
class Chain:
    index: int
    n_residues: int
    n_atoms: int
    n_segments: int
    n_frags: int
    centroid: FloatArray
    coords: FloatArray
    residue_numbers: IntArray
    segment_numbers: IntArray
    segment_adjacency: BoolArray
    frag_numbers: IntArray
    weights: FloatArray
    is_ill: bool
    residue_id_labels: tuple[str, ...] | None = None
    segment_links: list["SegmentLink"] = field(default_factory=list)
    scores: FloatArray | None = None
    solutions: FloatArray | None = None
    rigid_score_mean: float | None = None
    rigid_score_std: float | None = None
    segment_score_means: FloatArray | None = None
    segment_score_stds: FloatArray | None = None


@dataclass(slots=True)
class SegmentLink:
    left_segment: int
    right_segment: int
    left_anchor_index: int
    right_anchor_index: int
    residue_id_a: str
    residue_id_b: str


@dataclass(slots=True)
class DomainLink:
    chain_index: int
    residue_id_a: str
    residue_id_b: str
    segment_a: int
    segment_b: int


@dataclass(slots=True)
class PDBModel:
    path: Path
    lines: list[str]
    chains: list[Chain]
    total_residues: int
    links: list[DomainLink] = field(default_factory=list)


def _segment_anchor_indices(chain: Chain, left_ids: IntArray, right_ids: IntArray) -> tuple[int, int]:
    left_residues = chain.residue_numbers[left_ids]
    right_residues = chain.residue_numbers[right_ids]
    if float(np.mean(left_residues)) <= float(np.mean(right_residues)):
        left_anchor_idx = int(left_ids[np.argmax(left_residues)])
        right_anchor_idx = int(right_ids[np.argmin(right_residues)])
    else:
        left_anchor_idx = int(left_ids[np.argmin(left_residues)])
        right_anchor_idx = int(right_ids[np.argmax(right_residues)])
    return left_anchor_idx, right_anchor_idx


def segment_anchor_pairs(
    chain: Chain,
    segment_indices: dict[int, IntArray] | None = None,
) -> list[tuple[int, int, int, int]]:
    if chain.n_segments <= 1:
        return []

    if chain.segment_links:
        return [
            (
                int(link.left_segment),
                int(link.right_segment),
                int(link.left_anchor_index),
                int(link.right_anchor_index),
            )
            for link in chain.segment_links
            if 1 <= int(link.left_segment) <= chain.n_segments
            and 1 <= int(link.right_segment) <= chain.n_segments
            and int(link.left_anchor_index) >= 0
            and int(link.right_anchor_index) >= 0
            and int(link.left_anchor_index) < chain.n_atoms
            and int(link.right_anchor_index) < chain.n_atoms
        ]

    if segment_indices is None:
        segment_indices = {
            segment: np.flatnonzero(chain.segment_numbers == segment).astype(np.int32, copy=False)
            for segment in range(1, chain.n_segments + 1)
        }

    adjacency = chain.segment_adjacency
    if adjacency.shape != (chain.n_segments, chain.n_segments):
        adjacency = linear_segment_adjacency(chain.n_segments)

    pairs: list[tuple[int, int, int, int]] = []
    for left in range(1, chain.n_segments + 1):
        for right in range(left + 1, chain.n_segments + 1):
            if not adjacency[left - 1, right - 1]:
                continue
            left_ids = segment_indices.get(left)
            right_ids = segment_indices.get(right)
            if left_ids is None or right_ids is None or len(left_ids) == 0 or len(right_ids) == 0:
                continue
            left_anchor_idx, right_anchor_idx = _segment_anchor_indices(chain, left_ids, right_ids)
            pairs.append((left, right, left_anchor_idx, right_anchor_idx))
    return pairs
