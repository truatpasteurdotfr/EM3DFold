from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from scipy import fft, ndimage as ndi, optimize

from em3dfit.config import Params
from em3dfit.score import euler_to_matrix
from em3dfit.types import Chain
from em3dfit.utils import log_message as _base_log_message, stage_timer as _base_stage_timer

LOG_STAGE = "Rigid"
log_message = partial(_base_log_message, stage=LOG_STAGE)
stage_timer = partial(_base_stage_timer, stage=LOG_STAGE)
RIGID_TORCH_BATCH_SIZE_MAX = 128
RIGID_TORCH_BATCH_SIZE_FALLBACK = 16
RIGID_TORCH_BATCH_MEMORY_FRACTION = 0.55
RIGID_USE_TORCH_LOCAL_REFINE = False
FTMATCH_GRID_SCALE = 0.1
FFT_FRIENDLY_FACTORS = np.asarray(
    sorted(
        {
            (2**k1) * (3**k2) * (5**k3) * (7**k4)
            for k1 in range(8)
            for k2 in range(6)
            for k3 in range(5)
            for k4 in range(4)
        }
    ),
    dtype=np.int64,
)


@dataclass(slots=True)
class Pose:
    solution: np.ndarray
    score: float
    normalized_score: float | None = None


@dataclass(slots=True)
class SearchContext:
    centered_target: np.ndarray
    centered_chain: np.ndarray
    target_weights: np.ndarray
    chain_weights: np.ndarray
    centrioda: np.ndarray
    centriodb: np.ndarray
    flowera: np.ndarray
    nxyz: np.ndarray
    target_fft: np.ndarray | None
    refinement_grid: np.ndarray
    slowera: np.ndarray
    nxyz0: np.ndarray
    target_fft_torch: torch.Tensor | None = None
    centered_chain_torch: torch.Tensor | None = None
    chain_weights_torch: torch.Tensor | None = None
    flowera_torch: torch.Tensor | None = None
    refinement_grid_torch: torch.Tensor | None = None
    refinement_grid_volume_torch: torch.Tensor | None = None
    slowera_torch: torch.Tensor | None = None
    rigid_device: str = "cpu"


def generate_angle_set(angle_step: float) -> np.ndarray:
    if angle_step < 6.0:
        raise ValueError("angle_step must be >= 6.0 degrees")

    pi = math.pi
    dist = 2.0 * math.sin(pi * angle_step / 360.0)
    dtheta0 = angle_step * pi / 180.0
    dpsi = 2.0 * pi / (int((2.0 * pi - dtheta0 / 2.0) / dtheta0) + 1)
    dtheta = pi / (int((pi - dtheta0 / 2.0) / dtheta0) + 1)
    angles: list[tuple[float, float, float]] = []

    theta = 0.0
    while theta <= pi - dtheta + 1e-8:
        rz = max(math.sin(theta), 1e-6)
        dphi_raw = min(2.0 * math.asin(min(dist / (2.0 * rz), 1.0)), 2.0 * pi)
        dphi = 2.0 * pi / (int((2.0 * pi - dphi_raw / 2.0) / dphi_raw) + 1)
        phi = 0.0
        while phi <= 2.0 * pi - dphi / 2.0 + 1e-8:
            psi = 0.0
            while psi <= 2.0 * pi - dpsi / 2.0 + 1e-8:
                angles.append((psi, theta, phi))
                psi += dpsi
            phi += dphi
        theta += dtheta
    return np.asarray(angles, dtype=np.float32)


def _splat_points(points: np.ndarray, weights: np.ndarray, lower: np.ndarray, spacing: float, shape: np.ndarray) -> np.ndarray:
    grid = np.zeros(tuple(int(v) for v in shape), dtype=np.float32)
    ijk = np.rint((points - lower) / spacing).astype(np.int32)
    valid = np.all((ijk >= 0) & (ijk < shape), axis=1)
    ijk = ijk[valid]
    w = weights[valid]
    np.add.at(grid, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), w)
    return grid


def _build_smoothed_grid(points: np.ndarray, weights: np.ndarray, lower: np.ndarray, spacing: float, shape: np.ndarray, resol: float) -> np.ndarray:
    rsigma2 = (math.pi * spacing / (2.4 + 0.8 * resol)) ** 2
    sigma = 1.0 / math.sqrt(2.0 * rsigma2)
    truncate = math.sqrt(12.0)
    grid = _splat_points(points, weights, lower, spacing, shape)
    return ndi.gaussian_filter(grid, sigma=sigma, truncate=truncate, mode="constant").astype(np.float32, copy=False)


def _gaussian_kernel_1d_torch(rsigma2: float, device: torch.device) -> tuple[torch.Tensor, int]:
    sigma = 1.0 / math.sqrt(2.0 * rsigma2)
    truncate = math.sqrt(12.0)
    radius = max(1, int(truncate * sigma + 0.5))
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-rsigma2 * offsets.square())
    kernel /= torch.clamp(torch.sum(kernel), min=1e-12)
    return kernel, radius


def _conv3d_separable_torch(volume_zyx: torch.Tensor, kernel_1d: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_x = kernel_1d.view(1, 1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, 1, -1, 1)
    kernel_z = kernel_1d.view(1, 1, -1, 1, 1)
    out = F.conv3d(volume_zyx, kernel_x, padding=(0, 0, radius))
    out = F.conv3d(out, kernel_y, padding=(0, radius, 0))
    out = F.conv3d(out, kernel_z, padding=(radius, 0, 0))
    return out


def _as_torch_float32(values: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32)
    return torch.from_numpy(values.astype(np.float32, copy=False)).to(device=device)


def _euler_to_matrix_torch(angles: torch.Tensor) -> torch.Tensor:
    angles_t = angles.to(dtype=torch.float32)
    if angles_t.ndim == 1:
        angles_t = angles_t.unsqueeze(0)

    phi = angles_t[:, 0]
    the = angles_t[:, 1]
    psi = angles_t[:, 2]

    cpsi = torch.cos(psi)
    spsi = torch.sin(psi)
    cthe = torch.cos(the)
    sthe = torch.sin(the)
    cphi = torch.cos(phi)
    sphi = torch.sin(phi)

    row0 = torch.stack((cpsi * cphi - cthe * sphi * spsi, cpsi * sphi + cthe * cphi * spsi, spsi * sthe), dim=1)
    row1 = torch.stack((-spsi * cphi - cthe * sphi * cpsi, -spsi * sphi + cthe * cphi * cpsi, cpsi * sthe), dim=1)
    row2 = torch.stack((sthe * sphi, -sthe * cphi, cthe), dim=1)
    return torch.stack((row0, row1, row2), dim=1)


def _splat_points_torch(
    points: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    lower: np.ndarray | torch.Tensor,
    spacing: float,
    shape: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    shape_xyz = tuple(int(v) for v in shape)
    grid_size = shape_xyz[0] * shape_xyz[1] * shape_xyz[2]
    points_t = _as_torch_float32(points, device)
    weights_t = _as_torch_float32(weights, device)
    lower_t = _as_torch_float32(lower, device)

    single_batch = points_t.ndim == 2
    if single_batch:
        points_t = points_t.unsqueeze(0)
    if weights_t.ndim == 1:
        weights_t = weights_t.unsqueeze(0).expand(points_t.shape[0], -1)
    elif weights_t.ndim == 2 and weights_t.shape[0] == 1 and points_t.shape[0] > 1:
        weights_t = weights_t.expand(points_t.shape[0], -1)
    if lower_t.ndim == 1:
        lower_t = lower_t.unsqueeze(0).expand(points_t.shape[0], -1)

    batch_size = points_t.shape[0]
    grid = torch.zeros((batch_size, *shape_xyz), dtype=torch.float32, device=device)
    ijk = torch.round((points_t - lower_t[:, None, :]) / float(spacing)).to(dtype=torch.int64)
    shape_t = torch.tensor(shape_xyz, dtype=torch.int64, device=device)
    valid = torch.all((ijk >= 0) & (ijk < shape_t), dim=2)
    if not torch.any(valid):
        return grid.squeeze(0) if single_batch else grid

    linear_xyz = ijk[..., 0] * (shape_xyz[1] * shape_xyz[2]) + ijk[..., 1] * shape_xyz[2] + ijk[..., 2]
    batch_ids = torch.arange(batch_size, dtype=torch.int64, device=device).unsqueeze(1).expand_as(valid)
    linear = batch_ids[valid] * grid_size + linear_xyz[valid]
    grid.view(-1).scatter_add_(0, linear, weights_t[valid])
    return grid.squeeze(0) if single_batch else grid


def _build_smoothed_grid_torch(
    points: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    lower: np.ndarray | torch.Tensor,
    spacing: float,
    shape: np.ndarray,
    resol: float,
    device: torch.device,
) -> torch.Tensor:
    rsigma2 = (math.pi * spacing / (2.4 + 0.8 * resol)) ** 2
    kernel_1d, radius = _gaussian_kernel_1d_torch(rsigma2, device)
    grid_xyz = _splat_points_torch(points, weights, lower, spacing, shape, device)
    single_batch = grid_xyz.ndim == 3
    if single_batch:
        grid_xyz = grid_xyz.unsqueeze(0)
    volume_zyx = grid_xyz.permute(0, 3, 2, 1).contiguous().unsqueeze(1)
    smoothed_zyx = _conv3d_separable_torch(volume_zyx, kernel_1d, radius)
    smoothed_xyz = smoothed_zyx.squeeze(1).permute(0, 3, 2, 1).contiguous()
    return smoothed_xyz.squeeze(0) if single_batch else smoothed_xyz


def _rigid_device_name(params: Params) -> str:
    if params.backend == "scipy":
        return "cpu"
    if params.backend == "torch":
        if params.device == "cuda":
            return "cuda"
        return "torch-cpu"
    if params.device == "cpu":
        return "torch-cpu"
    if params.device == "cuda":
        return "cuda" if torch.cuda.is_available() else "torch-cpu"
    return "cuda" if torch.cuda.is_available() else "torch-cpu"


def _torch_device_from_backend(rigid_device: str) -> torch.device | None:
    if rigid_device == "cuda":
        return torch.device("cuda")
    if rigid_device == "torch-cpu":
        return torch.device("cpu")
    return None


def _available_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        return int(free_bytes)

    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None

    if hasattr(os, "sysconf"):
        page_size_key = "SC_PAGE_SIZE" if "SC_PAGE_SIZE" in os.sysconf_names else "SC_PAGESIZE"
        if "SC_AVPHYS_PAGES" in os.sysconf_names and page_size_key in os.sysconf_names:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf(page_size_key))
    return None


def _estimated_torch_coarse_bytes_per_rotation(nxyz: np.ndarray) -> int:
    nx, ny, nz = (int(v) for v in nxyz)
    voxels = nx * ny * nz
    fft_voxels = nx * ny * (nz // 2 + 1)
    return int(16 * voxels + 8 * fft_voxels)


def _choose_torch_batch_size_from_memory(
    nxyz: np.ndarray,
    available_bytes: int | None,
    max_batch_size: int = RIGID_TORCH_BATCH_SIZE_MAX,
) -> int:
    if available_bytes is None or available_bytes <= 0:
        return min(max_batch_size, RIGID_TORCH_BATCH_SIZE_FALLBACK)
    bytes_per_rotation = max(_estimated_torch_coarse_bytes_per_rotation(nxyz), 1)
    usable_bytes = max(int(available_bytes * RIGID_TORCH_BATCH_MEMORY_FRACTION), bytes_per_rotation)
    batch_size = usable_bytes // bytes_per_rotation
    return max(1, min(int(batch_size), max_batch_size))


def _choose_torch_batch_size(
    nxyz: np.ndarray,
    device: torch.device,
    max_batch_size: int = RIGID_TORCH_BATCH_SIZE_MAX,
) -> int:
    available_bytes = _available_memory_bytes(device)
    return _choose_torch_batch_size_from_memory(nxyz, available_bytes, max_batch_size=max_batch_size)


def _next_fft_friendly_shape(nxyz: np.ndarray) -> np.ndarray:
    target = np.asarray(nxyz, dtype=np.int32).copy()
    adjusted = np.empty(3, dtype=np.int32)
    for axis in range(3):
        idx = int(np.searchsorted(FFT_FRIENDLY_FACTORS, target[axis], side="left"))
        if idx >= len(FFT_FRIENDLY_FACTORS):
            adjusted[axis] = int(target[axis])
        else:
            adjusted[axis] = int(FFT_FRIENDLY_FACTORS[idx])
    return adjusted


def _grid_gaussian_params(spacing: float, resol: float) -> tuple[float, float, float]:
    rsigma2 = (math.pi * spacing / (2.4 + 0.8 * resol)) ** 2
    coeff = 10.0 * (rsigma2 / math.pi) ** 1.5
    bw2 = 6.0 / rsigma2
    return rsigma2, coeff, bw2


def _build_ftmatch_grid_cpu(
    points: np.ndarray,
    weights: np.ndarray,
    lower: np.ndarray,
    spacing: float,
    shape: np.ndarray,
    resol: float,
    negative: bool = False,
) -> np.ndarray:
    grid = np.zeros(tuple(int(v) for v in shape), dtype=np.float32)
    if points.size == 0:
        return grid

    rsigma2, coeff, bw2 = _grid_gaussian_params(spacing, resol)
    radius = int(math.ceil(math.sqrt(bw2)))
    offsets = np.asarray(
        [[dx, dy, dz] for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1) for dz in range(-radius, radius + 1)],
        dtype=np.int32,
    )
    pos = ((points - lower[None, :]) / float(spacing)).astype(np.float32, copy=False)
    base = np.floor(pos).astype(np.int32, copy=False)
    idx = base[:, None, :] + offsets[None, :, :]

    valid = np.all((idx >= 0) & (idx < shape[None, None, :]), axis=2)
    deltas = pos[:, None, :] - idx.astype(np.float32, copy=False)
    d2 = np.sum(deltas * deltas, axis=2)
    valid &= d2 < bw2
    if not np.any(valid):
        return grid

    probs = (FTMATCH_GRID_SCALE * coeff * np.exp(-rsigma2 * d2) * weights[:, None]).astype(np.float32, copy=False)
    if negative:
        probs = -probs
        reducer = np.minimum.at
    else:
        reducer = np.maximum.at

    linear = idx[..., 0] * (int(shape[1]) * int(shape[2])) + idx[..., 1] * int(shape[2]) + idx[..., 2]
    reducer(grid.reshape(-1), linear[valid], probs[valid])
    return grid


def _build_ftmatch_grid_torch(
    points: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    lower: np.ndarray | torch.Tensor,
    spacing: float,
    shape: np.ndarray,
    resol: float,
    device: torch.device,
    negative: bool = False,
) -> torch.Tensor:
    shape_xyz = tuple(int(v) for v in shape)
    grid = torch.zeros(shape_xyz[0] * shape_xyz[1] * shape_xyz[2], dtype=torch.float32, device=device)
    points_t = _as_torch_float32(points, device)
    weights_t = _as_torch_float32(weights, device)
    lower_t = _as_torch_float32(lower, device)
    if points_t.numel() == 0:
        return grid.view(shape_xyz)

    rsigma2, coeff, bw2 = _grid_gaussian_params(spacing, resol)
    radius = int(math.ceil(math.sqrt(bw2)))
    offsets = torch.tensor(
        [[dx, dy, dz] for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1) for dz in range(-radius, radius + 1)],
        dtype=torch.int64,
        device=device,
    )

    pos = (points_t - lower_t.unsqueeze(0)) / float(spacing)
    base = torch.floor(pos).to(dtype=torch.int64)
    idx = base[:, None, :] + offsets[None, :, :]
    shape_t = torch.tensor(shape_xyz, dtype=torch.int64, device=device)
    valid = torch.all((idx >= 0) & (idx < shape_t), dim=2)
    deltas = pos[:, None, :] - idx.to(dtype=torch.float32)
    d2 = torch.sum(deltas.square(), dim=2)
    valid &= d2 < float(bw2)
    if not torch.any(valid):
        return grid.view(shape_xyz)

    probs = (float(FTMATCH_GRID_SCALE * coeff) * torch.exp(-float(rsigma2) * d2) * weights_t[:, None]).to(
        dtype=torch.float32
    )
    if negative:
        probs = -probs
        reduce_name = "amin"
    else:
        reduce_name = "amax"

    linear = idx[..., 0] * (shape_xyz[1] * shape_xyz[2]) + idx[..., 1] * shape_xyz[2] + idx[..., 2]
    grid.scatter_reduce_(0, linear[valid], probs[valid], reduce=reduce_name, include_self=True)
    return grid.view(shape_xyz)


def _build_grid_cpu(
    points: np.ndarray,
    weights: np.ndarray,
    lower: np.ndarray,
    spacing: float,
    shape: np.ndarray,
    resol: float,
    method: str,
    negative: bool = False,
) -> np.ndarray:
    if method == "ftmatch":
        return _build_ftmatch_grid_cpu(points, weights, lower, spacing, shape, resol, negative=negative)
    if negative:
        return -_build_smoothed_grid(points, weights, lower, spacing, shape, resol)
    return _build_smoothed_grid(points, weights, lower, spacing, shape, resol)


def _build_grid_torch(
    points: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    lower: np.ndarray | torch.Tensor,
    spacing: float,
    shape: np.ndarray,
    resol: float,
    device: torch.device,
    method: str,
    negative: bool = False,
) -> torch.Tensor:
    if method == "ftmatch":
        return _build_ftmatch_grid_torch(points, weights, lower, spacing, shape, resol, device, negative=negative)
    grid = _build_smoothed_grid_torch(points, weights, lower, spacing, shape, resol, device)
    return -grid if negative else grid


def build_search_context(ldps: np.ndarray, ldps_dens: np.ndarray, chain: Chain, params: Params) -> SearchContext:
    lowera = ldps.min(axis=0)
    uppera = ldps.max(axis=0)
    lowerb = chain.coords.min(axis=0)
    upperb = chain.coords.max(axis=0)
    centrioda = ((lowera + uppera) / 2.0).astype(np.float32, copy=False)
    centriodb = ((lowerb + upperb) / 2.0).astype(np.float32, copy=False)

    centered_target = (ldps - centrioda).astype(np.float32, copy=False)
    centered_chain = (chain.coords - centriodb).astype(np.float32, copy=False)

    rsigma2 = (math.pi * params.fgrid / (2.4 + 0.8 * params.resol)) ** 2
    bw = math.sqrt(6.0 / rsigma2)
    rg = float(np.sqrt(np.max(np.sum(centered_chain * centered_chain, axis=1)))) + bw * params.fgrid

    flowera = lowera - centrioda - bw * params.fgrid
    fuppera = uppera - centrioda + bw * params.fgrid
    flowera = np.minimum(flowera, -rg)
    fuppera = np.maximum(fuppera, rg)
    nxyz = np.rint((fuppera - flowera) / params.fgrid + 2.0).astype(np.int32)
    if params.grid_method == "ftmatch":
        nxyz = _next_fft_friendly_shape(nxyz)

    rigid_device = _rigid_device_name(params)
    torch_device = _torch_device_from_backend(rigid_device)
    target_fft: np.ndarray | None = None
    target_fft_torch: torch.Tensor | None = None
    centered_chain_torch: torch.Tensor | None = None
    chain_weights_torch: torch.Tensor | None = None
    flowera_torch: torch.Tensor | None = None
    refinement_grid_torch: torch.Tensor | None = None
    refinement_grid_volume_torch: torch.Tensor | None = None
    slowera_torch: torch.Tensor | None = None
    if torch_device is None:
        target_grid = _build_grid_cpu(
            centered_target,
            ldps_dens,
            flowera.astype(np.float32, copy=False),
            params.fgrid,
            nxyz,
            params.resol,
            method=params.grid_method,
        )
        target_fft = np.conjugate(fft.rfftn(target_grid))

    rsigma2_s = (math.pi * params.sgrid / (2.4 + 0.8 * params.resol)) ** 2
    bw_s = math.sqrt(6.0 / rsigma2_s)
    slowera = lowera - centrioda - bw_s * params.sgrid
    suppera = uppera - centrioda + bw_s * params.sgrid
    nxyz0 = np.rint((suppera - slowera) / params.sgrid + 2.0).astype(np.int32)
    refinement_grid = _build_grid_cpu(
        centered_target,
        ldps_dens,
        slowera.astype(np.float32, copy=False),
        params.sgrid,
        nxyz0,
        params.resol,
        method=params.grid_method,
        negative=True,
    )
    if torch_device is not None:
        target_grid_torch = _build_grid_torch(
            centered_target,
            ldps_dens,
            flowera.astype(np.float32, copy=False),
            params.fgrid,
            nxyz,
            params.resol,
            torch_device,
            method=params.grid_method,
        )
        target_fft_torch = torch.conj(torch.fft.rfftn(target_grid_torch))
        centered_chain_torch = _as_torch_float32(centered_chain, torch_device)
        chain_weights_torch = _as_torch_float32(chain.weights, torch_device)
        flowera_torch = _as_torch_float32(flowera.astype(np.float32, copy=False), torch_device)
        refinement_grid_torch = _as_torch_float32(refinement_grid, torch_device)
        refinement_grid_volume_torch = refinement_grid_torch.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)
        slowera_torch = _as_torch_float32(slowera.astype(np.float32, copy=False), torch_device)

    return SearchContext(
        centered_target=centered_target,
        centered_chain=centered_chain,
        target_weights=ldps_dens.astype(np.float32, copy=False),
        chain_weights=chain.weights.astype(np.float32, copy=False),
        centrioda=centrioda,
        centriodb=centriodb,
        flowera=flowera.astype(np.float32, copy=False),
        nxyz=nxyz,
        target_fft=target_fft,
        refinement_grid=refinement_grid,
        slowera=slowera.astype(np.float32, copy=False),
        nxyz0=nxyz0,
        target_fft_torch=target_fft_torch,
        centered_chain_torch=centered_chain_torch,
        chain_weights_torch=chain_weights_torch,
        flowera_torch=flowera_torch,
        refinement_grid_torch=refinement_grid_torch,
        refinement_grid_volume_torch=refinement_grid_volume_torch,
        slowera_torch=slowera_torch,
        rigid_device=rigid_device,
    )


def build_initial_ldp_search_grid(
    ldps: np.ndarray,
    ldps_dens: np.ndarray,
    params: Params,
) -> tuple[np.ndarray, np.ndarray]:
    if params.grid_method == "ftmatch":
        grid, origin, _backend = build_initial_ldp_search_grid_ftmatch(ldps, ldps_dens, params)
        return grid, origin

    lowera = ldps.min(axis=0)
    uppera = ldps.max(axis=0)
    centrioda = ((lowera + uppera) / 2.0).astype(np.float32, copy=False)
    centered_target = (ldps - centrioda).astype(np.float32, copy=False)

    rsigma2 = (math.pi * params.fgrid / (2.4 + 0.8 * params.resol)) ** 2
    bw = math.sqrt(6.0 / rsigma2)
    flowera = lowera - centrioda - bw * params.fgrid
    fuppera = uppera - centrioda + bw * params.fgrid
    nxyz = np.rint((fuppera - flowera) / params.fgrid + 2.0).astype(np.int32)
    target_grid = _build_smoothed_grid(centered_target, ldps_dens, flowera, params.fgrid, nxyz, params.resol)
    origin = (centrioda + flowera).astype(np.float32, copy=False)
    return target_grid, origin


def build_initial_ldp_search_grid_ftmatch(
    ldps: np.ndarray,
    ldps_dens: np.ndarray,
    params: Params,
) -> tuple[np.ndarray, np.ndarray, str]:
    lowera = ldps.min(axis=0).astype(np.float32, copy=False)
    uppera = ldps.max(axis=0).astype(np.float32, copy=False)
    centrioda = ((lowera + uppera) / 2.0).astype(np.float32, copy=False)
    centered_target = (ldps - centrioda).astype(np.float32, copy=False)

    rsigma2, _coeff, bw2 = _grid_gaussian_params(params.fgrid, params.resol)
    bw = math.sqrt(bw2)
    flowera = lowera - centrioda - bw * params.fgrid
    fuppera = uppera - centrioda + bw * params.fgrid
    nxyz = np.rint((fuppera - flowera) / params.fgrid + 2.0).astype(np.int32)
    nxyz = _next_fft_friendly_shape(nxyz)

    rigid_device = _rigid_device_name(params)
    torch_device = _torch_device_from_backend(rigid_device)
    if torch_device is None:
        grid = _build_ftmatch_grid_cpu(centered_target, ldps_dens, flowera.astype(np.float32, copy=False), params.fgrid, nxyz, params.resol)
        backend = "cpu"
    else:
        grid = _build_ftmatch_grid_torch(
            centered_target,
            ldps_dens,
            flowera.astype(np.float32, copy=False),
            params.fgrid,
            nxyz,
            params.resol,
            torch_device,
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        backend = rigid_device
    origin = (centrioda + flowera).astype(np.float32, copy=False)
    return grid, origin, backend


def _top_translations(corr: np.ndarray, ntrans: int) -> list[tuple[np.ndarray, float]]:
    n_keep = min(max(1, ntrans), corr.size)
    flat = corr.ravel()
    indices = np.argpartition(flat, n_keep - 1)[:n_keep]
    ranked = indices[np.argsort(flat[indices])]
    poses: list[tuple[np.ndarray, float]] = []
    for idx in ranked:
        trans = np.asarray(np.unravel_index(int(idx), corr.shape), dtype=np.int32)
        poses.append((trans, float(flat[idx])))
    return poses


def _top_translations_torch(corr: torch.Tensor, ntrans: int) -> list[tuple[np.ndarray, float]]:
    n_keep = min(max(1, ntrans), int(corr.numel()))
    flat = corr.reshape(-1)
    values, indices = torch.topk(-flat, k=n_keep)
    scores = (-values).detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy()
    shape = tuple(int(v) for v in corr.shape)
    poses: list[tuple[np.ndarray, float]] = []
    for flat_idx, score in zip(indices_np.tolist(), scores.tolist(), strict=True):
        trans = np.asarray(np.unravel_index(int(flat_idx), shape), dtype=np.int32)
        poses.append((trans, float(score)))
    return poses


def _top_translations_torch_batch(corr: torch.Tensor, ntrans: int) -> tuple[np.ndarray, np.ndarray]:
    if corr.ndim != 4:
        raise ValueError("Expected batched correlation tensor with shape (batch, x, y, z)")
    n_keep = min(max(1, ntrans), int(corr.shape[1] * corr.shape[2] * corr.shape[3]))
    flat = corr.reshape(corr.shape[0], -1)
    values, indices = torch.topk(-flat, k=n_keep, dim=1)
    return indices.detach().cpu().numpy(), (-values).detach().cpu().numpy()


def _translation_from_fft_index(trans: np.ndarray, nxyz: np.ndarray, spacing: float) -> np.ndarray:
    shift = trans.astype(np.int32, copy=True)
    half = nxyz // 2
    for axis in range(3):
        if shift[axis] > half[axis]:
            shift[axis] -= int(nxyz[axis])
    return (-shift.astype(np.float32) * spacing).astype(np.float32, copy=False)


def _coarse_candidates_cpu(
    context: SearchContext,
    angles: np.ndarray,
    params: Params,
) -> tuple[list[np.ndarray], list[float]]:
    coarse_candidates: list[np.ndarray] = []
    coarse_scores: list[float] = []
    for angle in angles:
        rot = euler_to_matrix(angle)
        rotated = (context.centered_chain @ rot.T).astype(np.float32, copy=False)
        ligand_grid = -_build_smoothed_grid(
            rotated,
            context.chain_weights,
            context.flowera,
            params.fgrid,
            context.nxyz,
            params.resol,
        )
        corr = fft.irfftn(
            context.target_fft * fft.rfftn(ligand_grid),
            s=tuple(int(v) for v in context.nxyz),
        ).real.astype(np.float32, copy=False)
        for trans_idx, score in _top_translations(corr, params.ntrans):
            solution = np.zeros(6, dtype=np.float32)
            solution[:3] = angle
            solution[3:6] = _translation_from_fft_index(trans_idx, context.nxyz, params.fgrid)
            coarse_candidates.append(solution)
            coarse_scores.append(score)
    return coarse_candidates, coarse_scores


def _coarse_candidates_torch(
    context: SearchContext,
    angles: np.ndarray,
    params: Params,
    batch_size: int,
) -> tuple[list[np.ndarray], list[float]]:
    if context.target_fft_torch is None or context.centered_chain_torch is None:
        return _coarse_candidates_cpu(context, angles, params)

    device = _torch_device_from_backend(context.rigid_device)
    if device is None or context.chain_weights_torch is None or context.flowera_torch is None:
        return _coarse_candidates_cpu(context, angles, params)

    nxyz_shape = tuple(int(v) for v in context.nxyz)
    coarse_candidates: list[np.ndarray] = []
    coarse_scores: list[float] = []

    angles_t = _as_torch_float32(angles, device)
    chain_points_t = context.centered_chain_torch
    target_fft_t = context.target_fft_torch
    chain_weights_t = context.chain_weights_torch
    flowera_t = context.flowera_torch

    with torch.no_grad():
        for start in range(0, len(angles), batch_size):
            batch_angles_t = angles_t[start : start + batch_size]
            rot_mats = _euler_to_matrix_torch(batch_angles_t)
            rotated = torch.matmul(chain_points_t.unsqueeze(0), rot_mats.transpose(1, 2))
            ligand_grid = -_build_smoothed_grid_torch(
                rotated,
                chain_weights_t,
                flowera_t,
                params.fgrid,
                context.nxyz,
                params.resol,
                device,
            )
            corr = torch.fft.irfftn(
                target_fft_t.unsqueeze(0) * torch.fft.rfftn(ligand_grid, dim=(-3, -2, -1)),
                s=nxyz_shape,
                dim=(-3, -2, -1),
            ).real
            indices_np, scores_np = _top_translations_torch_batch(corr, params.ntrans)
            shape = tuple(int(v) for v in corr.shape[1:])
            batch_angles_np = batch_angles_t.detach().cpu().numpy()
            for batch_idx, angle in enumerate(batch_angles_np):
                for flat_idx, score in zip(indices_np[batch_idx].tolist(), scores_np[batch_idx].tolist(), strict=True):
                    trans_idx = np.asarray(np.unravel_index(int(flat_idx), shape), dtype=np.int32)
                    solution = np.zeros(6, dtype=np.float32)
                    solution[:3] = angle
                    solution[3:6] = _translation_from_fft_index(trans_idx, context.nxyz, params.fgrid)
                    coarse_candidates.append(solution)
                    coarse_scores.append(float(score))
    return coarse_candidates, coarse_scores


def rigid_grid_score(solution_centered: np.ndarray, context: SearchContext, params: Params) -> float:
    if RIGID_USE_TORCH_LOCAL_REFINE and context.rigid_device == "cuda" and context.chain_weights_torch is not None:
        with torch.no_grad():
            vals_t = _rigid_grid_values_torch_batch(solution_centered, context, params)
            score_t = torch.sum(vals_t * context.chain_weights_torch.unsqueeze(0), dim=1)
        return float(score_t[0].item())
    vals = _rigid_grid_values(solution_centered, context, params)
    return float(np.sum(vals * context.chain_weights))


def _rigid_grid_values_torch_batch(
    solutions_centered: np.ndarray | torch.Tensor,
    context: SearchContext,
    params: Params,
) -> torch.Tensor:
    if (
        context.centered_chain_torch is None
        or context.refinement_grid_volume_torch is None
        or context.slowera_torch is None
    ):
        raise ValueError("Torch rigid grid evaluation requires torch search-context buffers")

    device = context.centered_chain_torch.device
    solutions_t = _as_torch_float32(solutions_centered, device)
    if solutions_t.ndim == 1:
        solutions_t = solutions_t.unsqueeze(0)

    rot_mats = _euler_to_matrix_torch(solutions_t[:, :3])
    moved = torch.matmul(context.centered_chain_torch.unsqueeze(0), rot_mats.transpose(1, 2)) + solutions_t[:, None, 3:6]
    sample = (moved - context.slowera_torch[None, None, :]) / float(params.sgrid)

    nx = max(int(context.nxyz0[0]) - 1, 1)
    ny = max(int(context.nxyz0[1]) - 1, 1)
    nz = max(int(context.nxyz0[2]) - 1, 1)
    grid = torch.empty((solutions_t.shape[0], moved.shape[1], 1, 1, 3), dtype=torch.float32, device=device)
    grid[..., 0] = (sample[..., 0] * (2.0 / nx) - 1.0).unsqueeze(-1).unsqueeze(-1)
    grid[..., 1] = (sample[..., 1] * (2.0 / ny) - 1.0).unsqueeze(-1).unsqueeze(-1)
    grid[..., 2] = (sample[..., 2] * (2.0 / nz) - 1.0).unsqueeze(-1).unsqueeze(-1)

    values = F.grid_sample(
        context.refinement_grid_volume_torch.expand(solutions_t.shape[0], -1, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return values[:, 0, :, 0, 0]


def _rigid_grid_values_torch(
    solution_centered: np.ndarray | torch.Tensor,
    context: SearchContext,
    params: Params,
) -> np.ndarray:
    with torch.no_grad():
        values = _rigid_grid_values_torch_batch(solution_centered, context, params)
    return values.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def _rigid_grid_values(solution_centered: np.ndarray, context: SearchContext, params: Params) -> np.ndarray:
    if RIGID_USE_TORCH_LOCAL_REFINE and context.rigid_device == "cuda" and context.refinement_grid_volume_torch is not None:
        return _rigid_grid_values_torch(solution_centered, context, params)
    rot = euler_to_matrix(solution_centered[:3])
    moved = (context.centered_chain @ rot.T) + solution_centered[3:6]
    sample = ((moved - context.slowera) / params.sgrid).T
    return ndi.map_coordinates(context.refinement_grid, sample, order=1, mode="constant", cval=0.0).astype(
        np.float32, copy=False
    )


def _refine_options(refine_method: str) -> dict[str, float | int]:
    if refine_method == "Powell":
        return {"maxiter": 60, "xtol": 0.1, "ftol": 0.1}
    if refine_method == "Nelder-Mead":
        return {"maxiter": 120, "xatol": 0.1, "fatol": 0.1}
    raise ValueError(f"Unsupported refine_method: {refine_method}")


def _store_segment_score_stats(
    chain: Chain,
    segment_score_rows: list[np.ndarray],
    params: Params,
) -> None:
    means = np.zeros(chain.n_segments, dtype=np.float32)
    stds = np.full(chain.n_segments, max(params.flexible_revert_min_std, 1e-3), dtype=np.float32)
    if not segment_score_rows:
        chain.segment_score_means = means
        chain.segment_score_stds = stds
        return

    values = np.asarray(segment_score_rows, dtype=np.float32)
    means[:] = np.mean(values, axis=0).astype(np.float32, copy=False)
    if values.shape[0] >= 2:
        stds[:] = np.maximum(np.std(values, axis=0, ddof=1), params.flexible_revert_min_std).astype(np.float32, copy=False)
    chain.segment_score_means = means
    chain.segment_score_stds = stds


def _store_rigid_score_stats(chain: Chain, refined: list[Pose], params: Params) -> None:
    if not refined:
        chain.rigid_score_mean = 0.0
        chain.rigid_score_std = max(params.flexible_revert_min_std, 1e-3)
        return
    values = np.asarray([pose.score for pose in refined], dtype=np.float32)
    chain.rigid_score_mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0
    chain.rigid_score_std = max(std, params.flexible_revert_min_std)


def _rigid_pose_zscore(score: float, chain: Chain, params: Params) -> float:
    mean = 0.0 if chain.rigid_score_mean is None else chain.rigid_score_mean
    std = max(
        params.flexible_revert_min_std,
        1e-6,
        0.0 if chain.rigid_score_std is None else chain.rigid_score_std,
    )
    return float((score - mean) / std)


def _retain_clustered_poses(
    clustered: list[Pose],
    chain: Chain,
    params: Params,
    rigid_score_cutoff: float,
    rigid_nleast: int,
) -> list[Pose]:
    if not clustered:
        return clustered

    retained = 0
    for idx, pose in enumerate(clustered, start=1):
        if _rigid_pose_zscore(pose.score, chain, params) < rigid_score_cutoff:
            retained = idx
        else:
            break

    if len(clustered) <= rigid_nleast:
        retained = len(clustered)
    else:
        retained = max(retained, rigid_nleast)
    retained = min(retained, len(clustered), params.ntop)
    return clustered[:retained]


def search_chain_poses(
    chain: Chain,
    ldps: np.ndarray,
    ldps_dens: np.ndarray,
    params: Params,
    rigid_score_cutoff: float | None = None,
    rigid_nleast: int | None = None,
) -> list[Pose]:
    with stage_timer(f"chain {chain.index:02d} search context"):
        context = build_search_context(ldps, ldps_dens, chain, params)
    with stage_timer(f"chain {chain.index:02d} angle set"):
        angles = generate_angle_set(params.angle_step)
    log_message(f"chain {chain.index:02d} trying {len(angles)} rotations")
    if params.device == "cuda" and context.rigid_device != "cuda":
        log_message(f"chain {chain.index:02d} requested cuda but falling back to {context.rigid_device}")
    log_message(f"chain {chain.index:02d} coarse rigid backend {context.rigid_device}")
    coarse_candidates: list[np.ndarray] = []
    coarse_scores: list[float] = []
    coarse_batch_size = 1
    if context.target_fft_torch is not None:
        torch_device = _torch_device_from_backend(context.rigid_device)
        if torch_device is not None:
            coarse_batch_size = _choose_torch_batch_size(context.nxyz, torch_device, max_batch_size=RIGID_TORCH_BATCH_SIZE_MAX)
            log_message(f"chain {chain.index:02d} coarse rigid batch size {coarse_batch_size}")

    with stage_timer(f"chain {chain.index:02d} coarse rigid search"):
        if context.target_fft_torch is not None:
            coarse_candidates, coarse_scores = _coarse_candidates_torch(context, angles, params, coarse_batch_size)
        else:
            coarse_candidates, coarse_scores = _coarse_candidates_cpu(context, angles, params)

    if not coarse_candidates:
        return [Pose(solution=np.zeros(6, dtype=np.float32), score=0.0)]

    order = np.argsort(np.asarray(coarse_scores, dtype=np.float32))
    n_refine = min(len(order), max(params.ntop * 4, params.ntrans))
    refined: list[Pose] = []
    segment_score_rows: list[np.ndarray] = []
    with stage_timer(f"chain {chain.index:02d} local refinement"):
        log_message(f"chain {chain.index:02d} local refine method {params.refine_method}")
        for idx in order[:n_refine]:
            x0 = coarse_candidates[int(idx)]
            result = optimize.minimize(
                rigid_grid_score,
                x0=x0,
                args=(context, params),
                method=params.refine_method,
                options=_refine_options(params.refine_method),
            )
            best = np.asarray(result.x, dtype=np.float32)
            atom_values = _rigid_grid_values(best, context, params)
            atom_scores = (atom_values * context.chain_weights).astype(np.float32, copy=False)
            segment_scores = np.bincount(
                chain.segment_numbers.astype(np.int32) - 1,
                weights=atom_scores,
                minlength=chain.n_segments,
            ).astype(np.float32, copy=False)
            score = float(np.sum(atom_scores))
            world = best.copy()
            world[3:6] = best[3:6] - context.centriodb + context.centrioda
            refined.append(Pose(solution=world, score=score))
            segment_score_rows.append(segment_scores)

    _store_rigid_score_stats(chain, refined, params)
    _store_segment_score_stats(chain, segment_score_rows, params)
    for pose in refined:
        pose.normalized_score = _rigid_pose_zscore(pose.score, chain, params)
    refined.sort(key=lambda pose: pose.score)

    clustered: list[Pose] = []
    clustered_coords: list[np.ndarray] = []
    for pose in refined:
        rot = euler_to_matrix(pose.solution[:3])
        moved = (chain.coords - chain.centroid) @ rot.T + chain.centroid + pose.solution[3:6]
        keep = True
        for existing in clustered_coords:
            rmsd = float(np.sqrt(np.mean(np.sum((existing - moved) ** 2, axis=1))))
            if rmsd < params.rmsdcut1:
                keep = False
                break
        if keep:
            clustered.append(pose)
            clustered_coords.append(moved)
        if len(clustered) >= params.ntop:
            break

    if not clustered:
        clustered.append(refined[0])

    cutoff = params.rigid_cutoff_score_early if rigid_score_cutoff is None else rigid_score_cutoff
    min_keep = params.rigid_nleast if rigid_nleast is None else rigid_nleast
    retained = _retain_clustered_poses(clustered, chain, params, cutoff, min_keep)
    log_message(
        f"chain {chain.index:02d}: kept {len(retained)}/{len(clustered)} rigid pose(s) "
        f"after z-score cutoff {cutoff:.2f}, nleast={min(min_keep, params.ntop)}"
    )
    return retained
