from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from em3dfit.types import MRCMap


def _read_i4(header: bytes, offset: int, count: int, endian: str) -> NDArray[np.int32]:
    return np.frombuffer(header, dtype=f"{endian}i4", count=count, offset=offset).copy()


def _read_f4(header: bytes, offset: int, count: int, endian: str) -> NDArray[np.float32]:
    return np.frombuffer(header, dtype=f"{endian}f4", count=count, offset=offset).copy()


def _detect_endian(header: bytes) -> str:
    if header[212] == 68:
        return "<"
    if header[212] == 17:
        return ">"
    return "<"


def read_mrc(path: str | Path) -> MRCMap:
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(1024)
        if len(header) != 1024:
            raise ValueError(f"{path} is not a valid MRC file.")

        endian = _detect_endian(header)
        ncrs = tuple(int(x) for x in _read_i4(header, 0, 3, endian))
        mode = int(_read_i4(header, 12, 1, endian)[0])
        ncrsstart = tuple(int(x) for x in _read_i4(header, 16, 3, endian))
        mxyz = tuple(int(x) for x in _read_i4(header, 28, 3, endian))
        cella = _read_f4(header, 40, 3, endian)
        cellb = _read_f4(header, 52, 3, endian)
        mapcrs = tuple(int(x) for x in _read_i4(header, 64, 3, endian))
        nsymbt = int(_read_i4(header, 92, 1, endian)[0])
        origin = _read_f4(header, 196, 3, endian)
        map_tag = header[208:212]

        if map_tag != b"MAP ":
            raise ValueError(f"{path} is not in MRC2014 format.")
        if mode != 2:
            raise ValueError(f"{path} stores map values in mode {mode}; only float32 mode 2 is supported.")

        if nsymbt > 0:
            handle.read(nsymbt)
        raw = handle.read(int(np.prod(ncrs)) * 4)
        values = np.frombuffer(raw, dtype=f"{endian}f4").copy()

    data = values.reshape(ncrs, order="F").astype(np.float32, copy=False)
    return MRCMap(
        path=path,
        data=data,
        ncrs=ncrs,
        ncrsstart=ncrsstart,
        mxyz=mxyz,
        cella=cella.astype(np.float32, copy=False),
        cellb=cellb.astype(np.float32, copy=False),
        mapcrs=mapcrs,
        origin=origin.astype(np.float32, copy=False),
        mode=mode,
    )


def normalize_mrc(mrc: MRCMap, apix: float) -> MRCMap:
    voxel_size = mrc.voxel_size
    sort = np.empty(3, dtype=np.int32)
    for idx, axis in enumerate(mrc.mapcrs):
        sort[axis - 1] = idx

    reordered = np.transpose(mrc.data, axes=tuple(int(x) for x in sort)).astype(np.float32, copy=False)
    ncrs = tuple(int(x) for x in np.asarray(mrc.ncrs, dtype=np.int32)[sort])
    ncrsstart = tuple(int(x) for x in np.asarray(mrc.ncrsstart, dtype=np.int32)[sort])
    mxyz = tuple(int(x) for x in np.asarray(mrc.mxyz, dtype=np.int32)[sort])
    voxel_sorted = voxel_size[sort].astype(np.float32, copy=False)
    origin = (mrc.origin + np.asarray(ncrsstart, dtype=np.float32) * voxel_sorted).astype(np.float32, copy=False)

    cella = (np.asarray(mxyz, dtype=np.float32) * voxel_sorted).astype(np.float32, copy=False)
    if apix != -1.0 and not np.allclose(voxel_sorted, apix, atol=1e-5):
        new_shape = tuple(int(np.floor(v * (n - 1) / apix) + 1) for v, n in zip(voxel_sorted, reordered.shape))
        zoom = tuple((n_new - 1) / max(n_old - 1, 1) for n_new, n_old in zip(new_shape, reordered.shape))
        reordered = ndi.zoom(reordered, zoom=zoom, order=3).astype(np.float32, copy=False)
        voxel_sorted = np.full(3, apix, dtype=np.float32)
        cella = np.asarray(mxyz, dtype=np.float32) * voxel_sorted
        ncrs = tuple(int(x) for x in reordered.shape)

    return MRCMap(
        path=mrc.path,
        data=reordered,
        ncrs=ncrs,
        ncrsstart=(0, 0, 0),
        mxyz=mxyz,
        cella=cella.astype(np.float32, copy=False),
        cellb=mrc.cellb,
        mapcrs=(1, 2, 3),
        origin=origin,
        mode=2,
    )


def write_mrc(
    path: str | Path,
    data: np.ndarray,
    voxel_size: float | tuple[float, float, float] | np.ndarray,
    origin: np.ndarray | tuple[float, float, float] | list[float] | None = None,
) -> None:
    path = Path(path)
    grid = np.asarray(data, dtype=np.float32)
    if grid.ndim != 3:
        raise ValueError("MRC output data must be a 3D float32 array.")

    if np.isscalar(voxel_size):
        voxel = np.full(3, float(voxel_size), dtype=np.float32)
    else:
        voxel = np.asarray(voxel_size, dtype=np.float32)
        if voxel.shape != (3,):
            raise ValueError("voxel_size must be a scalar or length-3 sequence.")

    if origin is None:
        origin_arr = np.zeros(3, dtype=np.float32)
    else:
        origin_arr = np.asarray(origin, dtype=np.float32)
        if origin_arr.shape != (3,):
            raise ValueError("origin must be length 3 when provided.")

    nxyz = np.asarray(grid.shape, dtype=np.int32)
    cella = (nxyz.astype(np.float32) * voxel).astype(np.float32, copy=False)
    stats_min = float(np.min(grid))
    stats_max = float(np.max(grid))
    stats_mean = float(np.mean(grid))

    header = bytearray(1024)
    header[0:12] = np.asarray(nxyz, dtype="<i4").tobytes()
    header[12:16] = np.asarray([2], dtype="<i4").tobytes()
    header[16:28] = np.asarray([0, 0, 0], dtype="<i4").tobytes()
    header[28:40] = np.asarray(nxyz, dtype="<i4").tobytes()
    header[40:52] = np.asarray(cella, dtype="<f4").tobytes()
    header[52:64] = np.asarray([90.0, 90.0, 90.0], dtype="<f4").tobytes()
    header[64:76] = np.asarray([1, 2, 3], dtype="<i4").tobytes()
    header[76:88] = np.asarray([stats_min, stats_max, stats_mean], dtype="<f4").tobytes()
    header[88:92] = np.asarray([0], dtype="<i4").tobytes()
    header[92:96] = np.asarray([0], dtype="<i4").tobytes()
    header[196:208] = np.asarray(origin_arr, dtype="<f4").tobytes()
    header[208:212] = b"MAP "
    header[212:216] = bytes((68, 65, 0, 0))
    header[216:220] = np.asarray([0.0], dtype="<f4").tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(np.asarray(grid, dtype="<f4").tobytes(order="F"))
