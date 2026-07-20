"""Small NumPy geometry helpers shared across EM3DFold pipelines."""

from __future__ import annotations

import numpy as np


def split_chain_to_frags(ca_coords: np.ndarray, threshold: float = 8.0) -> list[list[int]]:
    ca_coords = np.asarray(ca_coords)
    dists = np.linalg.norm(ca_coords[1:] - ca_coords[:-1], axis=1)
    breaks = np.where(dists > threshold)[0] + 1

    segments = []
    start = 0
    for end in breaks:
        segments.append(list(range(start, end)))
        start = end
    segments.append(list(range(start, len(ca_coords))))
    return segments


def distance(a, b):
    return np.linalg.norm(a - b)


def pairwise_distances(a, b):
    assert a.ndim == b.ndim
    assert a.shape[-1] == b.shape[-1]
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=-1), axis=-1))


def kabsch(P, Q):
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    assert len(P) == len(Q)
    assert len(P) > 0

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.T.dot(Q_centered)
    U, _S, VT = np.linalg.svd(H)
    R = U.dot(VT).T
    if np.linalg.det(R) < 0:
        VT[2, :] *= -1
        R = U.dot(VT).T
    t = centroid_Q - R.dot(centroid_P)
    return R, t


def kabsch_rmsd(P, Q):
    R, T = kabsch(P, Q)
    axes = tuple(range(R.ndim - 2)) + (-1, -2)
    rP = np.matmul(P, R.transpose(axes)) + T
    return np.sqrt(np.mean(np.sum((rP - Q) ** 2, axis=-1), axis=-1))


def kabschx(P, Q):
    assert P.shape == Q.shape
    assert len(P.shape) >= 2 and P.shape[-1] == 3

    centroid_P = np.mean(P, axis=-2, keepdims=True)
    centroid_Q = np.mean(Q, axis=-2, keepdims=True)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    axes = tuple(range(P_centered.ndim - 2)) + (-1, -2)
    H = np.matmul(P_centered.transpose(axes), Q_centered)
    U, _S, VT = np.linalg.svd(H)
    R = np.matmul(U, VT)
    axes = tuple(range(R.ndim - 2)) + (-1, -2)
    R = R.transpose(axes)

    det = np.linalg.det(R)
    sign = np.where(det < 0.0, -1, 1)[..., None]
    VT[..., 2, :] *= sign
    R = np.matmul(U, VT)
    R = R.transpose(axes)
    t = centroid_Q - np.matmul(R, centroid_P.transpose(axes)).transpose(axes)
    return R, t


def kabschx_apply(P, R, t):
    axes = tuple(range(R.ndim - 2)) + (-1, -2)
    return np.matmul(P, R.transpose(axes)) + t


def kabschx_rmsd(P, Q):
    R, T = kabschx(P, Q)
    axes = tuple(range(R.ndim - 2)) + (-1, -2)
    rP = np.matmul(P, R.transpose(axes)) + T
    return np.sqrt(np.mean(np.sum((rP - Q) ** 2, axis=-1), axis=-1))


def apply(x, R, t):
    return kabschx_apply(x, R, t)
