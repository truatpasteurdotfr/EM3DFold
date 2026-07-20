from typing import Tuple

import numpy as np
from scipy.interpolate import interpn
from scipy.spatial import cKDTree


def sample_uniformly_on_sphere(sphere_radius: float, num_points: int) -> np.ndarray:
    points = np.random.randn(num_points, 3)
    points /= np.linalg.norm(points, axis=-1, ord=2, keepdims=True)
    return points * sphere_radius


def get_reference_gaussian_params(cryo_map) -> Tuple[float, float]:
    map_max = np.max(cryo_map.grid)
    map_min = np.min(cryo_map.grid)
    map_mean = np.mean(cryo_map.grid)
    map_std = np.std(cryo_map.grid)
    high_value = min(map_mean + 10 * map_std, map_max)
    low_value = max(map_mean - map_std, map_min)
    reference_gaussian_height = high_value - low_value
    reference_gaussian_offset = low_value
    return reference_gaussian_height, reference_gaussian_offset


def get_radial_points(
    atoms: np.ndarray, sphere_radius: float, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    radial_points = np.zeros((len(atoms), num_points, 3))
    point_exists = np.zeros((len(atoms), num_points), dtype=bool)
    kdtree = cKDTree(atoms)

    for num_try in range(100):
        atoms_left = ~np.all(point_exists, axis=-1)
        num_atoms_left = np.sum(atoms_left)
        sphere_points = sample_uniformly_on_sphere(
            sphere_radius, num_points * num_atoms_left
        ).reshape((num_atoms_left, num_points, 3))
        sphere_points += atoms[atoms_left, None]

        indices = kdtree.query(sphere_points, k=1, workers=4)[1]
        indices_that_match = indices == np.arange(len(atoms))[atoms_left, None]

        if num_try > 3:
            sort_idx_pe = np.argsort(~point_exists[atoms_left], axis=1)
            point_exists[atoms_left] = np.take_along_axis(
                point_exists[atoms_left], sort_idx_pe, axis=1
            )
            radial_points[atoms_left] = np.take_along_axis(
                radial_points[atoms_left], sort_idx_pe[..., None], axis=1
            )
            sort_idx_sp = np.argsort(indices_that_match, axis=1)
            indices_that_match = np.take_along_axis(
                indices_that_match, sort_idx_sp, axis=1
            )
            sphere_points = np.take_along_axis(
                sphere_points, sort_idx_sp[..., None], axis=1
            )

        idxs_to_update = np.nonzero(indices_that_match & ~point_exists[atoms_left])
        radial_points[
            np.nonzero(atoms_left)[0][idxs_to_update[0]], idxs_to_update[1]
        ] = sphere_points[idxs_to_update[0], idxs_to_update[1]]
        point_exists[
            np.nonzero(atoms_left)[0][idxs_to_update[0]], idxs_to_update[1]
        ] = True

        if np.all(point_exists):
            break

    return radial_points, point_exists


def interpolate_grid_at_points(points: np.ndarray, cryo_map) -> np.ndarray:
    x = np.arange(cryo_map.grid.shape[0])
    y = np.arange(cryo_map.grid.shape[1])
    z = np.arange(cryo_map.grid.shape[2])

    p = (points - cryo_map.global_origin[None]) / cryo_map.voxel_size
    p = np.flip(p, axis=-1)
    p = np.clip(p, 0.0, np.asarray(cryo_map.grid.shape) - 1)

    return interpn((x, y, z), cryo_map.grid, p)

