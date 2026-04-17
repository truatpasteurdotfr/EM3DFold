import numpy as np

from tqdm import tqdm

from em3dfold.utils.qscore.utils import (
    get_radial_points,
    get_reference_gaussian_params,
    interpolate_grid_at_points,
)

def calculate_q_score(
    atoms,
    map,
    ref_gaussian_width: float = 0.6,
    num_points: int = 8,
    epsilon: float = 1e-6,
) -> np.ndarray:
    num_atoms = len(atoms)
    ref_gaussian_height, ref_gaussian_offset = get_reference_gaussian_params(map)
    reference_gaussian_values = []
    map_values = []
    for R in tqdm(range(21)):
        R /= 10
        radial_points = get_radial_points(atoms, R, num_points)
        map_values_at_points = interpolate_grid_at_points(radial_points[0], map)
        map_values.append(map_values_at_points)
        ref_gaussian_value_at_R = ref_gaussian_height * np.exp(
            - 0.5 * (R/ref_gaussian_width)**2
        ) + ref_gaussian_offset
        reference_gaussian_values.append([ref_gaussian_value_at_R] * num_points)
    v_vector = np.stack(reference_gaussian_values, axis=1).reshape(1, -1)  # N x M
    u_vector = np.stack(map_values, axis=2).reshape(num_atoms, -1)  # A x N x M
    v_norm = v_vector - np.mean(v_vector, axis=1, keepdims=True)
    u_norm = u_vector - np.mean(u_vector, axis=1, keepdims=True)
    Q = np.sum(u_norm * v_norm, axis=-1) / np.sqrt(
            np.sum(u_norm * u_norm, axis=-1) * np.sum(v_norm * v_norm, axis=-1) + epsilon
    )
    return Q


