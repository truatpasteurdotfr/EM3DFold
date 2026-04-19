"""Grid interpolation helpers shared by template-guided utilities."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interpn


def grid_value_interp(points, grid, origin=None, vsize=None):
    if origin is None:
        origin = np.zeros(3, dtype=np.float32)
    if vsize is None:
        vsize = np.ones(3, dtype=np.float32)

    n0 = np.arange(grid.shape[0])
    n1 = np.arange(grid.shape[1])
    n2 = np.arange(grid.shape[2])
    p = (points - origin) / vsize
    p = np.flip(p, axis=-1)

    p[..., 0] = np.clip(p[..., 0], 0.0, grid.shape[0] - 1)
    p[..., 1] = np.clip(p[..., 1], 0.0, grid.shape[1] - 1)
    p[..., 2] = np.clip(p[..., 2], 0.0, grid.shape[2] - 1)

    values = interpn(
        (n0, n1, n2),
        grid,
        p,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return values.astype(np.float32)
