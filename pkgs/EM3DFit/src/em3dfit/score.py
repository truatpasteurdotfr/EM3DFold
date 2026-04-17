from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from em3dfit.config import Params
from em3dfit.types import Chain


def euler_to_matrix(angles: np.ndarray) -> np.ndarray:
    phi, the, psi = (float(v) for v in angles)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    cthe, sthe = math.cos(the), math.sin(the)
    cphi, sphi = math.cos(phi), math.sin(phi)
    return np.asarray(
        [
            [cpsi * cphi - cthe * sphi * spsi, cpsi * sphi + cthe * cphi * spsi, spsi * sthe],
            [-spsi * cphi - cthe * sphi * cpsi, -spsi * sphi + cthe * cphi * cpsi, cpsi * sthe],
            [sthe * sphi, -sthe * cphi, cthe],
        ],
        dtype=np.float32,
    )


def transform_points(points: np.ndarray, centroid: np.ndarray, solution: np.ndarray) -> np.ndarray:
    rot = euler_to_matrix(solution[:3])
    return (points - centroid) @ rot.T + centroid + solution[3:6]


def clash_score(coords_a: np.ndarray, coords_b: np.ndarray, params: Params) -> float:
    if coords_a.size == 0 or coords_b.size == 0:
        return 0.0
    rsigma2 = (math.pi / (2.4 + 0.8 * params.resol)) ** 2
    bw = math.sqrt(6.0 / rsigma2)
    tree = cKDTree(coords_b)
    dists, _ = tree.query(coords_a, distance_upper_bound=bw)
    valid = np.isfinite(dists)
    if not np.any(valid):
        return 0.0
    scores = np.zeros(len(coords_a), dtype=np.float32)
    d2s = np.maximum(dists[valid] - params.clash_dist, 0.0) ** 2
    scores[valid] = np.exp(-rsigma2 * d2s).astype(np.float32, copy=False)
    return float(np.mean(scores))


def score_chain_against_ldps(chain: Chain, ldps: np.ndarray, ldps_dens: np.ndarray, params: Params) -> np.ndarray:
    rsigma2 = (math.pi * params.sgrid / params.ldp_kernel_scale) ** 2
    coeff = 10.0 * (rsigma2 / math.pi) ** 1.5
    bw2 = 6.0 / rsigma2
    radius = math.sqrt(bw2) * params.sgrid

    tree = cKDTree(ldps)
    scores = np.zeros(chain.n_atoms, dtype=np.float32)

    rot_mats = []
    translations = []
    if chain.solutions is not None:
        for segment_idx in range(chain.n_segments):
            solution = chain.solutions[segment_idx]
            rot_mats.append(euler_to_matrix(solution[:3]))
            translations.append(solution[3:6].astype(np.float32, copy=False))
    else:
        for _ in range(chain.n_segments):
            rot_mats.append(np.eye(3, dtype=np.float32))
            translations.append(np.zeros(3, dtype=np.float32))

    for atom_idx, coord in enumerate(chain.coords):
        segment_idx = int(chain.segment_numbers[atom_idx]) - 1
        rot = rot_mats[segment_idx]
        trans = translations[segment_idx]
        moved = rot @ (coord - chain.centroid) + trans + chain.centroid
        hits = tree.query_ball_point(moved, r=radius)
        if not hits:
            continue
        delta = ldps[np.asarray(hits, dtype=np.int32)] - moved
        dist2_grid = np.sum(delta * delta, axis=1) / (params.sgrid**2)
        probs = coeff * np.exp(-rsigma2 * dist2_grid) * ldps_dens[hits]
        scores[atom_idx] = -float(np.max(probs)) * float(chain.weights[atom_idx])
    return scores
