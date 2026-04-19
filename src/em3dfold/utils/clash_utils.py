"""Clash utilities shared by template-guided and mainline pipelines."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numba import njit
from scipy.spatial import KDTree


def clash_ratio(a, b, r_clash=1.8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    assert a.ndim == 2 and b.ndim == 2
    atree = KDTree(a)
    btree = KDTree(b)

    idxs_a_in_b = atree.query_ball_point(b, r=r_clash)
    idxs_b_in_a = btree.query_ball_point(a, r=r_clash)

    if len(idxs_a_in_b) > 0:
        idxs_a_in_b = np.concatenate(idxs_a_in_b, axis=0).astype(np.int32)
        idxs_a_in_b = np.unique(idxs_a_in_b)

    if len(idxs_b_in_a) > 0:
        idxs_b_in_a = np.concatenate(idxs_b_in_a, axis=0).astype(np.int32)
        idxs_b_in_a = np.unique(idxs_b_in_a)

    ra = len(idxs_a_in_b) / len(a)
    rb = len(idxs_b_in_a) / len(b)
    return ra, rb


def clash_flag(a, b, r_clash=1.8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    assert a.ndim == 2 and b.ndim == 2
    atree = KDTree(a)
    btree = KDTree(b)

    ia = np.zeros(len(a), dtype=np.int32)
    ib = np.zeros(len(b), dtype=np.int32)

    idxs_a_in_b = atree.query_ball_point(b, r=r_clash)
    idxs_b_in_a = btree.query_ball_point(a, r=r_clash)

    if len(idxs_a_in_b) > 0:
        idxs_a_in_b = np.concatenate(idxs_a_in_b, axis=0).astype(np.int32)
        idxs_a_in_b = np.unique(idxs_a_in_b)
        ia[idxs_a_in_b] = 1

    if len(idxs_b_in_a) > 0:
        idxs_b_in_a = np.concatenate(idxs_b_in_a, axis=0).astype(np.int32)
        idxs_b_in_a = np.unique(idxs_b_in_a)
        ib[idxs_b_in_a] = 1

    return ia, ib


@njit
def _get_clash_kernel(
    crda: np.ndarray,
    densa: np.ndarray,
    crdb: np.ndarray,
    densb: np.ndarray,
    resol: float = 5.0,
    clash_dist: float = 1.5,
    compute_b_scores: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pi = np.pi
    na = crda.shape[0]
    nb = crdb.shape[0]

    rsigma2 = (pi / (2.4 + 0.8 * resol)) ** 2
    bw2 = 6.0 / rsigma2
    bw = np.sqrt(bw2)
    sqrt3bw = np.sqrt(3.0) * bw

    scoresa = np.zeros(na)
    mind2a = np.full(na, bw2)

    if compute_b_scores:
        scoresb = np.zeros(nb)
        mind2b = np.full(nb, bw2)
    else:
        scoresb = None

    for i in range(na):
        for j in range(nb):
            dens = densa[i] * densb[j]
            ftmp = np.abs(crda[i] - crdb[j])

            if np.max(ftmp) > bw or np.sum(ftmp) > sqrt3bw:
                continue

            d2 = np.sum(ftmp ** 2)
            d2s = max(np.sqrt(d2) - clash_dist, 0.0) ** 2

            if d2 < bw2:
                prob = np.exp(-rsigma2 * d2s)

                if d2 < mind2a[i]:
                    scoresa[i] = prob * dens
                    mind2a[i] = d2

                if compute_b_scores and d2 < mind2b[j]:
                    scoresb[j] = prob * dens
                    mind2b[j] = d2

    return scoresa, scoresb


def get_clash(
    crda: np.ndarray,
    densa: np.ndarray,
    crdb: np.ndarray,
    densb: np.ndarray,
    resol: float = 5.0,
    clash_dist: float = 1.0,
    return_b_scores: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert crda.shape[1] == 3, "crda must be (n, 3) array"
    assert crdb.shape[1] == 3, "crdb must be (m, 3) array"
    assert len(crda) == len(densa), "crda and densa must have same length"
    assert len(crdb) == len(densb), "crdb and densb must have same length"

    scoresa, scoresb = _get_clash_kernel(
        crda, densa, crdb, densb, resol, clash_dist, return_b_scores
    )
    return (scoresa, scoresb) if return_b_scores else (scoresa, None)
