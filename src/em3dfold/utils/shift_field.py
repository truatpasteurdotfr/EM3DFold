"""Smooth local shift-field helpers."""

from __future__ import annotations

import numpy as np


def dist2(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    assert a.ndim == b.ndim
    assert a.shape[-1] == b.shape[-1]
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)


def create_shift_field(fixing, moving, u=15.0):
    assert fixing.shape == moving.shape
    shift_vectors = moving - fixing

    def smoothed_shift_field(x):
        assert x.ndim == 2
        x = np.asarray(x, dtype=np.float32)
        weights = np.exp(-dist2(x, fixing) / u ** 2)
        weighted_shifts = weights @ shift_vectors
        sum_weights = np.sum(weights, axis=-1)
        return weighted_shifts / (sum_weights[..., None] + 1e-6)

    return smoothed_shift_field
