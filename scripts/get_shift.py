#!/usr/bin/env python
"""Shift MA output coordinates onto the original input map.

This script prints that translation in Angstroms and can apply it to an mmCIF:

    corrected_origin = map_origin + nstart * voxel_size - cubic_pad_shift * voxel_size

Add the printed vector to every MA output coordinate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _axis_reorder(mapc: int, mapr: int, maps: int) -> np.ndarray:
    """Return EM3DFold/MA-style CRS-to-XYZ ordering."""
    mapcrs = np.asarray([mapc, mapr, maps], dtype=np.int32) - 1
    if sorted(mapcrs.tolist()) != [0, 1, 2]:
        raise ValueError(f"Unsupported MRC axis order: mapc/mapr/maps = {mapc}/{mapr}/{maps}")

    order = np.asarray([0, 1, 2], dtype=np.int32)
    for i in range(3):
        order[mapcrs[i]] = i
    return order


def _ma_loaded_shape(data_shape: tuple[int, int, int], order: np.ndarray) -> np.ndarray:
    """Shape after ModelAngelo/EM3DFold transpose, in grid axis order z, y, x."""
    return np.asarray(data_shape, dtype=np.int64)[2 - order[::-1]]


def _ma_cubic_shift_zyx(shape_zyx: np.ndarray) -> tuple[np.ndarray, int]:
    """Replicate MA's make_cubic behavior without loading map data."""
    side = int(max(int(np.max(shape_zyx)), 128))
    side += side % 2
    if np.all(shape_zyx == side):
        return np.zeros(3, dtype=np.int64), side
    shift = np.asarray([side, side, side], dtype=np.int64) // 2 - shape_zyx // 2
    return shift, side


def compute_compensation(map_path: Path) -> dict[str, Any]:
    try:
        import mrcfile
    except ImportError as exc:
        raise SystemExit(
            "This script requires the Python package 'mrcfile'. "
            "Install the EM3DFold environment dependencies first."
        ) from exc

    with mrcfile.open(map_path, mode="r", header_only=True) as mrc:
        header = mrc.header
        voxel_size_xyz = np.asarray(
            [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float64
        )
        origin_xyz = np.asarray(
            [header.origin.x, header.origin.y, header.origin.z], dtype=np.float64
        )
        ncrsstart = np.asarray(
            [header.nxstart, header.nystart, header.nzstart], dtype=np.float64
        )
        order = _axis_reorder(int(header.mapc), int(header.mapr), int(header.maps))
        nxyzstart = ncrsstart[order]

        data_shape = tuple(int(v) for v in mrc.data.shape)
        loaded_shape_zyx = _ma_loaded_shape(data_shape, order)

    if np.any(voxel_size_xyz <= 0):
        raise ValueError(f"Invalid voxel size in {map_path}: {voxel_size_xyz.tolist()}")

    pad_shift_zyx, cubic_side = _ma_cubic_shift_zyx(loaded_shape_zyx)
    pad_shift_xyz = pad_shift_zyx[::-1].astype(np.float64)
    map_origin_xyz = origin_xyz + nxyzstart * voxel_size_xyz
    compensation_xyz = map_origin_xyz - pad_shift_xyz * voxel_size_xyz

    return {
        "map": str(map_path),
        "voxel_size_xyz": voxel_size_xyz.tolist(),
        "header_origin_xyz": origin_xyz.tolist(),
        "nxyzstart": nxyzstart.tolist(),
        "map_origin_xyz": map_origin_xyz.tolist(),
        "ma_loaded_shape_zyx": loaded_shape_zyx.tolist(),
        "ma_cubic_side": cubic_side,
        "ma_pad_shift_zyx": pad_shift_zyx.tolist(),
        "ma_pad_shift_xyz": pad_shift_xyz.tolist(),
        "compensation_xyz": compensation_xyz.tolist(),
    }


def _default_output_path(structure_path: Path) -> Path:
    suffix = structure_path.suffix
    stem = structure_path.with_suffix("")
    return stem.with_name(f"{stem.name}_shifted{suffix or '.cif'}")


def shift_cif(input_path: Path, output_path: Path, shift_xyz: np.ndarray) -> None:
    try:
        from Bio.PDB import MMCIFIO, MMCIFParser
    except ImportError as exc:
        raise SystemExit(
            "This script requires Biopython to shift mmCIF coordinates. "
            "Install the EM3DFold environment dependencies first."
        ) from exc

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(input_path.stem, str(input_path))
    shift = np.asarray(shift_xyz, dtype=np.float64)
    for atom in structure.get_atoms():
        atom.coord = atom.coord + shift

    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the XYZ Angstrom translation to add to ModelAngelo output "
            "coordinates, then optionally write a shifted mmCIF."
        )
    )
    parser.add_argument("map", type=Path, help="Input density map used by ModelAngelo.")
    parser.add_argument(
        "cif",
        type=Path,
        nargs="?",
        help="ModelAngelo output mmCIF to shift onto the input map.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Shifted mmCIF output path. Defaults to '<input>_shifted.cif'.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    args = parser.parse_args()

    result = compute_compensation(args.map)
    output_path = None
    if args.cif is not None:
        output_path = args.output if args.output is not None else _default_output_path(args.cif)
        shift_cif(args.cif, output_path, np.asarray(result["compensation_xyz"], dtype=np.float64))
        result["input_cif"] = str(args.cif)
        result["shifted_cif"] = str(output_path)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    dx, dy, dz = result["compensation_xyz"]
    print("Add this XYZ translation to ModelAngelo output coordinates (Angstrom):")
    print(f"{dx:.6f} {dy:.6f} {dz:.6f}")
    print()
    print("Details:")
    print(f"  map origin xyz:          {result['map_origin_xyz']}")
    print(f"  ModelAngelo pad xyz:     {result['ma_pad_shift_xyz']} voxels")
    print(f"  voxel size xyz:          {result['voxel_size_xyz']}")
    if output_path is not None:
        print(f"  shifted cif:             {output_path}")


if __name__ == "__main__":
    main()
