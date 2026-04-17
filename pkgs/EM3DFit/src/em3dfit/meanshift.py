from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from em3dfit.config import Params
from em3dfit.types import MRCMap
from em3dfit.utils import log_message as _base_log_message, stage_timer as _base_stage_timer

LOG_STAGE = "Meanshift"
log_message = partial(_base_log_message, stage=LOG_STAGE)
stage_timer = partial(_base_stage_timer, stage=LOG_STAGE)


def map_to_grid_points(mrc: MRCMap, threshold: float) -> np.ndarray:
    valid = np.argwhere(mrc.data > threshold)
    if valid.size == 0:
        raise ValueError("No valid grid points were found above the current threshold.")
    return valid.astype(np.float32, copy=False)


def _normalize_densities(densities: np.ndarray) -> np.ndarray:
    if len(densities) == 0:
        return np.zeros(0, dtype=np.float32)
    densities = densities.astype(np.float32, copy=False)
    min_dens = float(np.min(densities))
    max_dens = float(np.max(densities))
    if max_dens - min_dens < 1e-8:
        return np.full(len(densities), 2.0, dtype=np.float32)
    normalized = ((densities - min_dens) / (max_dens - min_dens)).astype(np.float32, copy=False)
    return (normalized + 1.0).astype(np.float32, copy=False)


def _gaussian_kernel_1d(rsigma2: float, device: torch.device) -> tuple[torch.Tensor, int]:
    bw = math.sqrt(6.0 / rsigma2)
    radius = max(1, int(math.ceil(bw)))
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-rsigma2 * offsets.square())
    return kernel, radius


def _conv3d_separable(volume: torch.Tensor, kernel_1d: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_x = kernel_1d.view(1, 1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, 1, -1, 1)
    kernel_z = kernel_1d.view(1, 1, -1, 1, 1)
    out = F.conv3d(volume, kernel_x, padding=(0, 0, radius))
    out = F.conv3d(out, kernel_y, padding=(0, radius, 0))
    out = F.conv3d(out, kernel_z, padding=(radius, 0, 0))
    return out


def _sample_torch(field: torch.Tensor, positions_zyx: torch.Tensor, shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    nz, ny, nx = shape_zyx
    z = 2.0 * positions_zyx[:, 0] / max(nz - 1, 1) - 1.0
    y = 2.0 * positions_zyx[:, 1] / max(ny - 1, 1) - 1.0
    x = 2.0 * positions_zyx[:, 2] / max(nx - 1, 1) - 1.0
    grid = torch.stack((x, y, z), dim=-1).view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.view(-1)


def _mean_shift_torch(mrc: MRCMap, grid_points: np.ndarray, params: Params, device_name: str) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(mrc.data, 0.0).astype(np.float32, copy=False)
    device = torch.device(device_name)

    volume_xyz = torch.from_numpy(positive).to(device=device)
    volume = volume_xyz.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)
    nz, ny, nx = volume.shape[-3:]

    rsigma2 = (math.pi * params.apix / params.ldp_kernel_scale) ** 2
    kernel_1d, radius = _gaussian_kernel_1d(rsigma2, device)

    xs = torch.arange(nx, device=device, dtype=torch.float32).view(1, 1, 1, 1, nx)
    ys = torch.arange(ny, device=device, dtype=torch.float32).view(1, 1, 1, ny, 1)
    zs = torch.arange(nz, device=device, dtype=torch.float32).view(1, 1, nz, 1, 1)

    smooth0 = _conv3d_separable(volume, kernel_1d, radius)
    smoothx = _conv3d_separable(volume * xs, kernel_1d, radius)
    smoothy = _conv3d_separable(volume * ys, kernel_1d, radius)
    smoothz = _conv3d_separable(volume * zs, kernel_1d, radius)

    positions = torch.from_numpy(grid_points.astype(np.float32, copy=False)).to(device=device)
    start = positions.clone()
    shift2_limit = (params.rshift / params.apix) ** 2

    for _ in range(params.max_shift_iterations):
        sample_zyx = torch.stack((positions[:, 2], positions[:, 1], positions[:, 0]), dim=1)
        denom = torch.clamp(_sample_torch(smooth0, sample_zyx, (nz, ny, nx)), min=1e-8)
        after = torch.stack(
            (
                _sample_torch(smoothx, sample_zyx, (nz, ny, nx)) / denom,
                _sample_torch(smoothy, sample_zyx, (nz, ny, nx)) / denom,
                _sample_torch(smoothz, sample_zyx, (nz, ny, nx)) / denom,
            ),
            dim=1,
        )
        step2 = torch.sum((after - positions) ** 2, dim=1)
        total2 = torch.sum((after - start) ** 2, dim=1)
        positions = after
        if torch.all((step2 < 1e-3) | (total2 > shift2_limit)):
            break

    sample_zyx = torch.stack((positions[:, 2], positions[:, 1], positions[:, 0]), dim=1)
    densities = _sample_torch(smooth0, sample_zyx, (nz, ny, nx))
    return positions.detach().cpu().numpy(), densities.detach().cpu().numpy()


def _mean_shift_scipy(mrc: MRCMap, grid_points: np.ndarray, params: Params) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(mrc.data, 0.0).astype(np.float32, copy=False)
    rsigma2 = (math.pi * params.apix / params.ldp_kernel_scale) ** 2
    sigma = 1.0 / math.sqrt(2.0 * rsigma2)
    truncate = math.sqrt(12.0)

    coords = np.indices(positive.shape, dtype=np.float32)
    smooth0 = ndi.gaussian_filter(positive, sigma=sigma, truncate=truncate, mode="constant")
    smoothx = ndi.gaussian_filter(positive * coords[0], sigma=sigma, truncate=truncate, mode="constant")
    smoothy = ndi.gaussian_filter(positive * coords[1], sigma=sigma, truncate=truncate, mode="constant")
    smoothz = ndi.gaussian_filter(positive * coords[2], sigma=sigma, truncate=truncate, mode="constant")

    positions = grid_points.astype(np.float32, copy=True)
    start = positions.copy()
    shift2_limit = (params.rshift / params.apix) ** 2

    for _ in range(params.max_shift_iterations):
        denom = np.maximum(ndi.map_coordinates(smooth0, positions.T, order=1, mode="nearest"), 1e-8)
        after = np.column_stack(
            [
                ndi.map_coordinates(smoothx, positions.T, order=1, mode="nearest") / denom,
                ndi.map_coordinates(smoothy, positions.T, order=1, mode="nearest") / denom,
                ndi.map_coordinates(smoothz, positions.T, order=1, mode="nearest") / denom,
            ]
        ).astype(np.float32, copy=False)
        step2 = np.sum((after - positions) ** 2, axis=1)
        total2 = np.sum((after - start) ** 2, axis=1)
        positions = after
        if np.all((step2 < 1e-3) | (total2 > shift2_limit)):
            break

    densities = ndi.map_coordinates(smooth0, positions.T, order=1, mode="nearest").astype(np.float32, copy=False)
    return positions, densities


def _merge_modes(points: np.ndarray, densities: np.ndarray, params: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("Mean-shift produced no candidate points.")

    normalized_densities = _normalize_densities(densities)
    valid = normalized_densities >= params.filter_fraction

    candidates = points[valid]
    candidate_dens = normalized_densities[valid]
    if len(candidates) == 0:
        raise ValueError("No candidate points survived the current filter threshold.")

    if len(candidates) == 1:
        membership = np.zeros(len(points), dtype=np.int32)
        return (
            candidates.astype(np.float32, copy=False),
            candidate_dens.astype(np.float32, copy=False),
            membership,
        )

    merge_radius = params.rmerge / params.apix
    tree = cKDTree(candidates)
    order = np.argsort(-candidate_dens)
    alive = np.ones(len(candidates), dtype=bool)
    centers_idx: list[int] = []
    for idx in order:
        if not alive[idx]:
            continue
        centers_idx.append(int(idx))
        neighbors = np.asarray(tree.query_ball_point(candidates[idx], r=merge_radius), dtype=np.int32)
        alive[neighbors] = False

    center_ids = np.asarray(centers_idx, dtype=np.int32)
    centers = candidates[center_ids]
    center_densities = candidate_dens[center_ids]
    member_tree = cKDTree(centers)
    membership = member_tree.query(points)[1].astype(np.int32, copy=False)
    return centers.astype(np.float32, copy=False), center_densities.astype(np.float32, copy=False), membership


def extract_ldps(mrc: MRCMap, params: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with stage_timer("grid point selection"):
        grid_points = map_to_grid_points(mrc, params.threshold)
    log_message(f"valid grid points: {len(grid_points)}")

    backend = params.backend
    if backend == "auto":
        backend = "torch" if torch.cuda.is_available() else "scipy"

    if backend == "torch":
        device = params.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log_message(f"mean-shift backend: torch ({device})")
        with stage_timer("mean-shift"):
            shifted, shifted_dens = _mean_shift_torch(mrc, grid_points, params, device)
    elif backend == "scipy":
        log_message("mean-shift backend: scipy")
        with stage_timer("mean-shift"):
            shifted, shifted_dens = _mean_shift_scipy(mrc, grid_points, params)
    else:
        raise ValueError(f"Unsupported backend: {params.backend}")
    shifted_dens = _normalize_densities(shifted_dens)

    with stage_timer("mode merging"):
        centers, center_densities, membership = _merge_modes(shifted, shifted_dens, params)
    ldps = centers * params.apix + mrc.origin.astype(np.float32, copy=False)
    return ldps.astype(np.float32, copy=False), center_densities, membership
