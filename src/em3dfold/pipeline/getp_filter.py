import os
import time
import numpy as np
from scipy.spatial import cKDTree

try:
    from numba import njit
except Exception:
    njit = None

from em3dfold.pipeline import meanshift
from em3dfold.utils.cryo_utils import parse_map, enlarge_grid

def write_p(
    filename,
    coords,
    dens=None,
    atom_name="CA",
    res_name="GLY",
    chain_id="A",
    element=None,
):
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if dens is None:
        dens = np.array([1.0] * len(coords))
    atom_name = atom_name.strip()
    res_name = res_name.strip()
    chain_id = (chain_id or "A")[0]
    if element is None:
        element = atom_name.replace("'", "").strip()[0]
    atom_name_pdb = f"{atom_name:>4s}"[:4]
    res_name_pdb = f"{res_name:>3s}"[:3]
    element_pdb = f"{element:>2s}"[:2]
    with open(filename, 'w') as f:
        for k, (coord, d) in enumerate(zip(coords, dens)):
            f.write("ATOM  {:>5d} {:4s} {:3s} {}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:>6.2f}{:>6.2f}          {:>2s}\n".format(
                k+1 if k+1 <= 99999 else 99999,
                atom_name_pdb,
                res_name_pdb,
                chain_id,
                k+1 if k+1 <= 9999 else 9999,
                coord[0],
                coord[1],
                coord[2],
                d,
                d,
                element_pdb,
            ))
            f.write("TER\n")

def get_lattice_meshgrid_np(shape, no_shift=False):
    linspace = np.linspace(
        0.5 if not no_shift else 0, shape - (0.5 if not no_shift else 1), shape,
    )
    mesh = np.stack(np.meshgrid(linspace, linspace, linspace, indexing="ij"), axis=-1,)
    return mesh

def _pack_neighborhoods(neighborhoods):
    n = len(neighborhoods)
    lengths = np.empty((n,), dtype=np.int32)
    total = 0
    for i, neigh in enumerate(neighborhoods):
        neigh_len = len(neigh)
        lengths[i] = neigh_len
        total += neigh_len

    offsets = np.empty((n + 1,), dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    flat = np.empty((total,), dtype=np.int32)
    cursor = 0
    for neigh in neighborhoods:
        neigh_len = len(neigh)
        if neigh_len > 0:
            flat[cursor:cursor + neigh_len] = neigh
            cursor += neigh_len
    return offsets, flat


if njit is not None:
    @njit(cache=True)
    def _prune_round_numba(points, probs, offsets, flat_neighbors):
        new_points = points.copy()
        updated_probs = probs.copy()

        for idx in range(len(offsets) - 1):
            start = offsets[idx]
            end = offsets[idx + 1]
            if end - start <= 1:
                continue

            prob_sum = 0.0
            keep_idx = -1
            keep_prob = -1.0
            x = 0.0
            y = 0.0
            z = 0.0

            for j in range(start, end):
                neighbor_idx = flat_neighbors[j]
                prob = updated_probs[neighbor_idx]
                prob_sum += prob
                if prob > keep_prob:
                    keep_prob = prob
                    keep_idx = neighbor_idx
                x += prob * points[neighbor_idx, 0]
                y += prob * points[neighbor_idx, 1]
                z += prob * points[neighbor_idx, 2]

            if prob_sum <= 0.0:
                continue

            for j in range(start, end):
                updated_probs[flat_neighbors[j]] = 0.0
            updated_probs[keep_idx] = prob_sum
            new_points[keep_idx, 0] = x / prob_sum
            new_points[keep_idx, 1] = y / prob_sum
            new_points[keep_idx, 2] = z / prob_sum

        keep_mask = updated_probs > 0.0
        return new_points[keep_mask], updated_probs[keep_mask]


def _prune_round_python(points, probs, offsets, flat_neighbors):
    new_points = np.copy(points)
    updated_probs = np.copy(probs)
    for idx in range(len(offsets) - 1):
        start = offsets[idx]
        end = offsets[idx + 1]
        if end - start <= 1:
            continue
        selection = flat_neighbors[start:end]
        selected_probs = updated_probs[selection]
        prob_sum = np.sum(selected_probs)
        if prob_sum <= 0:
            continue
        keep_local_idx = int(np.argmax(selected_probs))
        keep_idx = int(selection[keep_local_idx])
        new_points[keep_idx] = (
            np.sum(selected_probs[:, None] * points[selection], axis=0)
            / prob_sum
        )
        updated_probs[selection] = 0
        updated_probs[keep_idx] = prob_sum

    keep_mask = updated_probs > 0
    return new_points[keep_mask], updated_probs[keep_mask]


if njit is not None:
    @njit(cache=True)
    def _merge_keep_mask_numba(sorted_indices, offsets, flat_neighbors, n_points):
        kept = np.ones((n_points,), dtype=np.bool_)
        for order_idx in range(sorted_indices.shape[0]):
            point_idx = sorted_indices[order_idx]
            if not kept[point_idx]:
                continue
            start = offsets[point_idx]
            end = offsets[point_idx + 1]
            for j in range(start, end):
                neighbor_idx = flat_neighbors[j]
                if neighbor_idx != point_idx:
                    kept[neighbor_idx] = False
        return kept


    @njit(cache=True)
    def _greedy_select_indices_numba(order, offsets, flat_neighbors, n_points, max_points):
        suppressed = np.zeros((n_points,), dtype=np.bool_)
        selected = np.empty((order.shape[0],), dtype=np.int32)
        count = 0

        for order_idx in range(order.shape[0]):
            point_idx = order[order_idx]
            if suppressed[point_idx]:
                continue

            selected[count] = point_idx
            count += 1

            start = offsets[point_idx]
            end = offsets[point_idx + 1]
            for j in range(start, end):
                suppressed[flat_neighbors[j]] = True

            if max_points >= 0 and count >= max_points:
                break

        return selected[:count]


def _merge_keep_mask_python(sorted_indices, offsets, flat_neighbors, n_points):
    kept = np.ones((n_points,), dtype=bool)
    for point_idx in sorted_indices:
        if not kept[point_idx]:
            continue
        start = offsets[point_idx]
        end = offsets[point_idx + 1]
        neighbors = flat_neighbors[start:end]
        kept[neighbors[neighbors != point_idx]] = False
    return kept


def _greedy_select_indices_python(order, offsets, flat_neighbors, n_points, max_points):
    suppressed = np.zeros((n_points,), dtype=bool)
    selected_indices = []
    for point_idx in order:
        if suppressed[point_idx]:
            continue
        selected_indices.append(int(point_idx))
        start = offsets[point_idx]
        end = offsets[point_idx + 1]
        suppressed[flat_neighbors[start:end]] = True
        if max_points >= 0 and len(selected_indices) >= max_points:
            break
    return np.asarray(selected_indices, dtype=np.int32)


def grid_to_points(
    grid,
    threshold,
    neighbour_distance_threshold,
    prune_distance=1.1,
    return_timing=False,
):
    timing = {}
    lattice = np.flip(get_lattice_meshgrid_np(grid.shape[-1], no_shift=True), -1)

    t_stage = time.perf_counter()
    selected = grid > threshold
    output_points_before_pruning = np.copy(lattice[selected, :].reshape(-1, 3))

    points = lattice[selected, :].reshape(-1, 3)
    probs = grid[selected]
    timing["g2p_grid_select_points"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    for _ in range(3):
        kdtree = cKDTree(points)
        neighborhoods = kdtree.query_ball_tree(kdtree, r=prune_distance)
        offsets, flat_neighbors = _pack_neighborhoods(neighborhoods)
        if njit is not None:
            points, probs = _prune_round_numba(points, probs, offsets, flat_neighbors)
        else:
            points, probs = _prune_round_python(points, probs, offsets, flat_neighbors)
    timing["g2p_grid_prune_rounds"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    if len(points) > 0 and neighbour_distance_threshold > 0.0:
        kdtree = cKDTree(points)
        distances, _ = kdtree.query(points, k=2)
        points = points[distances[:, 1] <= neighbour_distance_threshold].reshape(-1, 3)
    timing["g2p_grid_neighbor_filter"] = time.perf_counter() - t_stage

    output_points = points
    if return_timing:
        return output_points, output_points_before_pruning, timing
    return output_points, output_points_before_pruning


def merge(coord, dens, d=1.0):
    print("# Merge distance = {:.4f}".format(d))

    n = len(coord)
    if n == 0:
        return

    tree = cKDTree(coord)
    neighborhoods = tree.query_ball_tree(tree, r=d)
    offsets, flat_neighbors = _pack_neighborhoods(neighborhoods)

    sorted_indices = np.argsort(dens)[::-1].astype(np.int32)
    if njit is not None:
        kept = _merge_keep_mask_numba(sorted_indices, offsets, flat_neighbors, n)
    else:
        kept = _merge_keep_mask_python(sorted_indices, offsets, flat_neighbors, n)

    return coord[kept], dens[kept]


def filter_connected_components(
    coords,
    dens,
    link_distance,
    min_component_size=0,
    min_fraction_largest=0.0,
):
    coords = np.asarray(coords, dtype=np.float32)
    dens = np.asarray(dens, dtype=np.float32)
    if len(coords) == 0:
        return coords, dens, {
            "num_components": 0,
            "largest_component": 0,
            "kept_components": 0,
            "points_before": 0,
            "points_after": 0,
            "effective_min_size": 0,
        }

    if link_distance <= 0.0 or (min_component_size <= 0 and min_fraction_largest <= 0.0):
        return coords, dens, {
            "num_components": 1,
            "largest_component": int(len(coords)),
            "kept_components": 1,
            "points_before": int(len(coords)),
            "points_after": int(len(coords)),
            "effective_min_size": 0,
        }

    tree = cKDTree(coords)
    neighborhoods = tree.query_ball_tree(tree, r=float(link_distance))
    visited = np.zeros((len(coords),), dtype=bool)
    components = []

    for start_idx in range(len(coords)):
        if visited[start_idx]:
            continue
        stack = [start_idx]
        visited[start_idx] = True
        component = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            for neighbor_idx in neighborhoods[idx]:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    stack.append(neighbor_idx)
        components.append(np.asarray(component, dtype=np.int32))

    component_sizes = np.asarray([len(component) for component in components], dtype=np.int32)
    largest_component = int(component_sizes.max()) if len(component_sizes) > 0 else 0
    min_fraction_size = int(np.ceil(float(largest_component) * float(min_fraction_largest))) if largest_component > 0 and min_fraction_largest > 0.0 else 0
    effective_min_size = max(int(min_component_size), int(min_fraction_size))

    keep_mask = np.zeros((len(coords),), dtype=bool)
    kept_components = 0
    for component in components:
        if len(component) < effective_min_size:
            continue
        keep_mask[component] = True
        kept_components += 1

    if kept_components == 0:
        keep_mask[:] = True
        kept_components = len(components)
        effective_min_size = 0

    filtered_coords = coords[keep_mask]
    filtered_dens = dens[keep_mask]
    stats = {
        "num_components": int(len(components)),
        "largest_component": int(largest_component),
        "kept_components": int(kept_components),
        "points_before": int(len(coords)),
        "points_after": int(len(filtered_coords)),
        "effective_min_size": int(effective_min_size),
    }
    return filtered_coords, filtered_dens, stats


def _coords_to_grid_xyz(coords, origin, voxel_size):
    coords = np.asarray(coords, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    return (coords - origin[None, :]) / voxel_size[None, :]


def sample_density_nearest(grid, coords, origin, voxel_size):
    if len(coords) == 0:
        return np.zeros((0,), dtype=np.float32)

    grid_xyz = _coords_to_grid_xyz(coords, origin, voxel_size)
    grid_idx = np.rint(grid_xyz).astype(np.int32)
    nxyz = np.asarray(grid.shape[::-1], dtype=np.int32)
    valid = np.all((grid_idx >= 0) & (grid_idx < nxyz[None, :]), axis=-1)

    sampled = np.zeros((len(coords),), dtype=np.float32)
    valid_idx = grid_idx[valid]
    sampled[valid] = grid[
        valid_idx[:, 2],
        valid_idx[:, 1],
        valid_idx[:, 0],
    ]
    return sampled


if njit is not None:
    @njit(cache=True)
    def _refine_points_to_density_centroid_numba(
        coords,
        grid,
        origin,
        voxel_size,
        nxyz,
        radius_grid,
        radius,
        n_iter,
    ):
        refined = coords.copy()
        radius_sq = radius * radius

        for _ in range(n_iter):
            for i in range(refined.shape[0]):
                coord_x = refined[i, 0]
                coord_y = refined[i, 1]
                coord_z = refined[i, 2]

                center_x = int(np.rint((coord_x - origin[0]) / voxel_size[0]))
                center_y = int(np.rint((coord_y - origin[1]) / voxel_size[1]))
                center_z = int(np.rint((coord_z - origin[2]) / voxel_size[2]))

                lower_x = max(center_x - radius_grid[0], 0)
                lower_y = max(center_y - radius_grid[1], 0)
                lower_z = max(center_z - radius_grid[2], 0)
                upper_x = min(center_x + radius_grid[0] + 1, nxyz[0])
                upper_y = min(center_y + radius_grid[1] + 1, nxyz[1])
                upper_z = min(center_z + radius_grid[2] + 1, nxyz[2])

                if lower_x >= upper_x or lower_y >= upper_y or lower_z >= upper_z:
                    continue

                weight_sum = 0.0
                weighted_x = 0.0
                weighted_y = 0.0
                weighted_z = 0.0

                for x_idx in range(lower_x, upper_x):
                    world_x = origin[0] + x_idx * voxel_size[0]
                    dx = world_x - coord_x
                    dx_sq = dx * dx
                    for y_idx in range(lower_y, upper_y):
                        world_y = origin[1] + y_idx * voxel_size[1]
                        dy = world_y - coord_y
                        dy_sq = dy * dy
                        for z_idx in range(lower_z, upper_z):
                            world_z = origin[2] + z_idx * voxel_size[2]
                            dz = world_z - coord_z
                            dist_sq = dx_sq + dy_sq + dz * dz
                            if dist_sq > radius_sq:
                                continue

                            weight = grid[z_idx, y_idx, x_idx]
                            if weight < 0.0:
                                weight = 0.0
                            weight_sum += weight
                            weighted_x += world_x * weight
                            weighted_y += world_y * weight
                            weighted_z += world_z * weight

                if weight_sum <= 1e-8:
                    continue

                refined[i, 0] = weighted_x / (weight_sum + 1e-8)
                refined[i, 1] = weighted_y / (weight_sum + 1e-8)
                refined[i, 2] = weighted_z / (weight_sum + 1e-8)

        return refined


def _refine_points_to_density_centroid_python(
    coords,
    grid,
    origin,
    voxel_size,
    nxyz,
    radius_grid,
    radius,
    n_iter,
):
    refined = np.copy(coords)

    for _ in range(n_iter):
        for i, coord in enumerate(refined):
            center_idx = np.rint(_coords_to_grid_xyz(coord[None, :], origin, voxel_size)[0]).astype(np.int32)
            lower = np.maximum(center_idx - radius_grid, 0)
            upper = np.minimum(center_idx + radius_grid + 1, nxyz)
            if np.any(lower >= upper):
                continue

            x_idx = np.arange(lower[0], upper[0], dtype=np.int32)
            y_idx = np.arange(lower[1], upper[1], dtype=np.int32)
            z_idx = np.arange(lower[2], upper[2], dtype=np.int32)
            mesh_xyz = np.stack(np.meshgrid(x_idx, y_idx, z_idx, indexing="ij"), axis=-1).reshape(-1, 3)
            world_xyz = origin[None, :] + mesh_xyz.astype(np.float32) * voxel_size[None, :]
            distance = np.linalg.norm(world_xyz - coord[None, :], axis=-1)
            within = distance <= radius
            if not np.any(within):
                continue

            valid_xyz = mesh_xyz[within]
            weights = grid[
                valid_xyz[:, 2],
                valid_xyz[:, 1],
                valid_xyz[:, 0],
            ]
            weights = np.clip(weights, a_min=0.0, a_max=None)
            if np.sum(weights) <= 1e-8:
                continue

            world_valid = world_xyz[within]
            refined[i] = np.sum(world_valid * weights[:, None], axis=0) / (np.sum(weights) + 1e-8)

    return refined


def refine_points_to_density_centroid(
    coords,
    grid,
    origin,
    voxel_size,
    radius=1.5,
    n_iter=2,
):
    coords = np.asarray(coords, dtype=np.float32)
    if len(coords) == 0 or radius <= 0.0 or n_iter <= 0:
        return coords

    grid = np.asarray(grid, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    nxyz = np.asarray(grid.shape[::-1], dtype=np.int32)
    radius_grid = np.maximum(
        1,
        np.ceil(radius / np.maximum(voxel_size, 1e-6)).astype(np.int32),
    )

    if njit is not None:
        return _refine_points_to_density_centroid_numba(
            coords,
            grid,
            origin,
            voxel_size,
            nxyz,
            radius_grid,
            float(radius),
            int(n_iter),
        )
    return _refine_points_to_density_centroid_python(
        coords,
        grid,
        origin,
        voxel_size,
        nxyz,
        radius_grid,
        radius,
        n_iter,
    )


def select_supplemental_g2p_points(
    getp_coords,
    g2p_coords,
    g2p_dens,
    cover_distance=1.75,
    supplement_merge_distance=1.5,
    max_points=None,
):
    getp_coords = np.asarray(getp_coords, dtype=np.float32)
    g2p_coords = np.asarray(g2p_coords, dtype=np.float32)
    g2p_dens = np.asarray(g2p_dens, dtype=np.float32)

    if len(g2p_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    keep = np.ones(len(g2p_coords), dtype=bool)
    if len(getp_coords) > 0:
        getp_tree = cKDTree(getp_coords)
        distance_to_getp, _ = getp_tree.query(g2p_coords, k=1)
        keep &= distance_to_getp > cover_distance

    candidate_coords = g2p_coords[keep]
    candidate_dens = g2p_dens[keep]
    if len(candidate_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = np.argsort(candidate_dens)[::-1].astype(np.int32)
    candidate_tree = cKDTree(candidate_coords)
    neighborhoods = candidate_tree.query_ball_tree(candidate_tree, r=supplement_merge_distance)
    offsets, flat_neighbors = _pack_neighborhoods(neighborhoods)
    max_points_int = -1 if max_points is None else int(max_points)
    if njit is not None:
        selected_indices = _greedy_select_indices_numba(order, offsets, flat_neighbors, len(candidate_coords), max_points_int)
    else:
        selected_indices = _greedy_select_indices_python(order, offsets, flat_neighbors, len(candidate_coords), max_points_int)

    if len(selected_indices) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return (
        candidate_coords[selected_indices],
        candidate_dens[selected_indices],
    )


def run_getp(
    map_dir,
    out_dir,
    getp_dir,
    pdb=None,
    init_coords=None,
    res=6.0,
    thresh=20,
    nt=4,
    filter=0.0,
    dmerge=1.0,
    rmax=10.0,
    device="auto",
    verbose=False,
    **kwargs,
):
    del getp_dir, nt
    if verbose:
        print(
            "# Running Python mean-shift getp replacement on {} -> {}".format(
                map_dir,
                out_dir,
            ),
            flush=True,
        )

    mrc = meanshift._load_mrc_map(map_dir)
    if init_coords is None and pdb is not None:
        init_coords = meanshift._read_pdb_points(pdb)

    resolved_device = meanshift._resolve_torch_device(device)
    backend = "torch" if str(resolved_device).startswith("cuda") else "scipy"
    params = meanshift.Params(
        threshold=float(thresh),
        resolution=float(res),
        rshift=float(rmax),
        rmerge=float(dmerge),
        filter_fraction=float(filter),
        backend=backend,
        device=str(resolved_device),
        max_shift_iterations=5000,
    )
    points, densities, _ = meanshift.extract_points(
        mrc,
        params,
        init_coords_xyz=init_coords,
    )
    if out_dir is not None:
        write_p(out_dir, points, densities)
    return points, densities


def main(args):
    stage_times = {}
    t_main_start = time.perf_counter()
    g2p_coords = np.zeros((0, 3), dtype=np.float32)
    g2p_dens = np.zeros((0,), dtype=np.float32)
    g2p_map = None
    g2p_origin = None
    g2p_vsize = None

    if args.run_g2p:
        t_stage = time.perf_counter()
        # Run grid to points
        t_sub = time.perf_counter()
        data, origin, nxyz, vsize = parse_map(args.map, False, None)
        stage_times["g2p_parse_map"] = time.perf_counter() - t_sub
        t_sub = time.perf_counter()
        data = enlarge_grid(data)
        stage_times["g2p_enlarge_grid"] = time.perf_counter() - t_sub

        t_sub = time.perf_counter()
        maximum = np.percentile(data, 99.999)
        data = np.clip(data, 0.0, maximum)
        data = data / (data.max() + 1e-6)
        stage_times["g2p_normalize"] = time.perf_counter() - t_sub

        t_sub = time.perf_counter()
        points, _, grid_to_points_timing = grid_to_points(
            data,
            args.ratio,
            float(getattr(args, "g2p_neighbor_distance_threshold", 6.0)),
            prune_distance=1.5,
            return_timing=True,
        )
        stage_times["g2p_grid_to_points"] = time.perf_counter() - t_sub
        stage_times.update(grid_to_points_timing)

        t_sub = time.perf_counter()
        g2p_coords = points + origin
        g2p_dens = sample_density_nearest(data, g2p_coords, origin, vsize)
        stage_times["g2p_sample_density"] = time.perf_counter() - t_sub
        g2p_map = data
        g2p_origin = origin
        g2p_vsize = vsize
        stage_times["g2p_seed"] = time.perf_counter() - t_stage
        print("# Done g2p n = {}".format(len(points)))
        print("# Stage time g2p_normalize = {:.4f}s".format(stage_times["g2p_normalize"]))
        print("# Stage time g2p_grid_select_points = {:.4f}s".format(stage_times["g2p_grid_select_points"]))
        print("# Stage time g2p_grid_prune_rounds = {:.4f}s".format(stage_times["g2p_grid_prune_rounds"]))
        print("# Stage time g2p_grid_neighbor_filter = {:.4f}s".format(stage_times["g2p_grid_neighbor_filter"]))
        print("# Stage time g2p_grid_to_points = {:.4f}s".format(stage_times["g2p_grid_to_points"]))
        print("# Stage time g2p_sample_density = {:.4f}s".format(stage_times["g2p_sample_density"]))
        print("# Stage time g2p_seed = {:.4f}s".format(stage_times["g2p_seed"]))

    # Run getp
    if args.run_getp:
        if len(g2p_coords) > 0:
            init_coords = g2p_coords
            init_pdb = None
        else:
            init_coords = None
            init_pdb = args.p

        t_stage = time.perf_counter()
        coords, dens = run_getp(
            map_dir=args.map,
            out_dir=None,
            getp_dir=args.getp,
            pdb=init_pdb,
            init_coords=init_coords,
            res=args.res,
            thresh=args.thresh,
            nt=args.nt,
            filter=args.filter,
            dmerge=args.dmerge / 4,
            rmax=args.rmax,
            device=args.device,
            verbose=True,
        )

        stage_times["getp_meanshift"] = time.perf_counter() - t_stage
        print("# Stage time getp_meanshift = {:.4f}s".format(stage_times["getp_meanshift"]))

        # Merge again
        print("# Before merging n = {}".format(len(coords)))
        t_stage = time.perf_counter()
        coords, dens = merge(coords, dens, args.dmerge)
        stage_times["merge_primary"] = time.perf_counter() - t_stage
        print("# After  merging n = {}".format(len(coords)))
        print("# Stage time merge_primary = {:.4f}s".format(stage_times["merge_primary"]))

        if args.run_g2p and args.fuse_g2p and len(g2p_coords) > 0:
            t_stage = time.perf_counter()
            supplemental_coords, supplemental_dens = select_supplemental_g2p_points(
                getp_coords=coords,
                g2p_coords=g2p_coords,
                g2p_dens=g2p_dens,
                cover_distance=args.g2p_cover_distance,
                supplement_merge_distance=args.g2p_supplement_merge_distance,
                max_points=args.g2p_max_supplements,
            )
            stage_times["g2p_select_supplements"] = time.perf_counter() - t_stage
            print("# Candidate g2p supplements n = {}".format(len(supplemental_coords)))
            print("# Stage time g2p_select_supplements = {:.4f}s".format(stage_times["g2p_select_supplements"]))

            if len(supplemental_coords) > 0 and args.g2p_refine_radius > 0.0:
                t_stage = time.perf_counter()
                supplemental_coords = refine_points_to_density_centroid(
                    supplemental_coords,
                    g2p_map,
                    g2p_origin,
                    g2p_vsize,
                    radius=args.g2p_refine_radius,
                    n_iter=args.g2p_refine_iters,
                )
                supplemental_dens = sample_density_nearest(
                    g2p_map,
                    supplemental_coords,
                    g2p_origin,
                    g2p_vsize,
                )
                stage_times["g2p_refine_supplements"] = time.perf_counter() - t_stage
                print("# Stage time g2p_refine_supplements = {:.4f}s".format(stage_times["g2p_refine_supplements"]))

            if len(supplemental_coords) > 0:
                coords = np.concatenate([coords, supplemental_coords], axis=0)
                dens = np.concatenate([dens, supplemental_dens], axis=0)
                print("# After adding g2p supplements n = {}".format(len(coords)))
                t_stage = time.perf_counter()
                coords, dens = merge(coords, dens, args.dmerge)
                stage_times["merge_with_supplements"] = time.perf_counter() - t_stage
                print("# After final merge with supplements n = {}".format(len(coords)))
                print("# Stage time merge_with_supplements = {:.4f}s".format(stage_times["merge_with_supplements"]))

        t_stage = time.perf_counter()
        coords, dens, cc_stats = filter_connected_components(
            coords,
            dens,
            link_distance=getattr(args, "component_link_distance", 0.0),
            min_component_size=getattr(args, "component_min_size", 0),
            min_fraction_largest=getattr(args, "component_min_fraction_largest", 0.0),
        )
        stage_times["connected_component_filter"] = time.perf_counter() - t_stage
        print("# Stage time connected_component_filter = {:.4f}s".format(stage_times["connected_component_filter"]))
        if getattr(args, "component_link_distance", 0.0) > 0.0 and (
            getattr(args, "component_min_size", 0) > 0
            or getattr(args, "component_min_fraction_largest", 0.0) > 0.0
        ):
            print(
                "# Connected-component filter: components={} largest={} kept={} min_size={} points={} -> {}".format(
                    cc_stats["num_components"],
                    cc_stats["largest_component"],
                    cc_stats["kept_components"],
                    cc_stats["effective_min_size"],
                    cc_stats["points_before"],
                    cc_stats["points_after"],
                )
            )

        final_dir = os.path.join(args.output, "merged.pdb")
        t_stage = time.perf_counter()
        write_p(
            final_dir,
            coords, dens,
            atom_name=args.atom_name,
            res_name=args.res_name,
            chain_id=args.chain_id,
            element=args.element,
        )
        stage_times["write_pdb"] = time.perf_counter() - t_stage
        print("# Final coords write to {}".format(final_dir))
        print("# Stage time write_pdb = {:.4f}s".format(stage_times["write_pdb"]))

    stage_times["total"] = time.perf_counter() - t_main_start
    print("# Stage timing summary")
    for key in [
        "g2p_parse_map",
        "g2p_enlarge_grid",
        "g2p_normalize",
        "g2p_grid_select_points",
        "g2p_grid_prune_rounds",
        "g2p_grid_neighbor_filter",
        "g2p_grid_to_points",
        "g2p_sample_density",
        "g2p_seed",
        "getp_meanshift",
        "merge_primary",
        "g2p_select_supplements",
        "g2p_refine_supplements",
        "merge_with_supplements",
        "connected_component_filter",
        "write_pdb",
        "total",
    ]:
        if key in stage_times:
            print("#   {:28s} {:.4f}s".format(key, stage_times[key]))

def add_args(parser):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--map", "-map", help="Input map", required=True)
    parser.add_argument("--device", default="cpu", help="Device to run on, CPU or GPU ID")
    # If input coords
    parser.add_argument("--p", "-p", help="Input coords in pdb format")
    # Others
    parser.add_argument("--output", "-o", help="Output directory", default='./')
    parser.add_argument("--getp", help="Getp binary directory", default="getp")
    parser.add_argument("--run-g2p", action='store_true')
    parser.add_argument("--run-getp", action='store_true')
    # Mean-Shift controls
    parser.add_argument("--ratio", type=float, default=0.05, help="Map threshold")
    parser.add_argument("--thresh", "-thresh", type=float, default=10.0, help="Map threshold")
    parser.add_argument("--res", "-res", type=float, default=6.0, help="Resolution")
    parser.add_argument("--nt", "-nt", type=int, default=4, help="Num of threads")
    parser.add_argument("--filter", "-filter", type=float, default=0.0, help="Filter thresh")
    parser.add_argument("--dmerge", "-dmerge", type=float, default=1.0, help="Merge distance")
    parser.add_argument("--rmax", "-rmax", type=float, default=5.0, help="Max shift distance")
    parser.set_defaults(fuse_g2p=True)
    parser.add_argument(
        "--no-fuse-g2p",
        dest="fuse_g2p",
        action="store_false",
        help="Disable g2p supplement fusion when both g2p and getp are used",
    )
    parser.add_argument(
        "--g2p-cover-distance",
        type=float,
        default=1.75,
        help="Drop g2p points within this distance to an existing getp point",
    )
    parser.add_argument(
        "--g2p-supplement-merge-distance",
        type=float,
        default=1.5,
        help="Greedy merge distance among retained g2p supplements",
    )
    parser.add_argument(
        "--g2p-refine-radius",
        type=float,
        default=1.5,
        help="Local density-centroid refinement radius for retained g2p supplements",
    )
    parser.add_argument(
        "--g2p-refine-iters",
        type=int,
        default=2,
        help="Number of local refinement iterations for retained g2p supplements",
    )
    parser.add_argument(
        "--g2p-max-supplements",
        type=int,
        default=None,
        help="Optional cap on the number of retained g2p supplement points",
    )
    parser.add_argument(
        "--g2p-neighbor-distance-threshold",
        type=float,
        default=6.0,
        help="Distance-based g2p seed filter threshold; disabled when <= 0",
    )
    parser.add_argument(
        "--component-link-distance",
        type=float,
        default=6.0,
        help="Graph edge distance for connected-component filtering; disabled when <= 0",
    )
    parser.add_argument(
        "--component-min-size",
        type=int,
        default=0,
        help="Drop connected components with fewer than this many points; disabled when <= 0",
    )
    parser.add_argument(
        "--component-min-fraction-largest",
        type=float,
        default=0.05,
        help="Drop connected components smaller than this fraction of the largest component; disabled when <= 0",
    )
    parser.add_argument("--atom-name", default="CA", help="Atom name used when writing output PDB points")
    parser.add_argument("--res-name", default="GLY", help="Residue name used when writing output PDB points")
    parser.add_argument("--chain-id", default="A", help="Chain ID used when writing output PDB points")
    parser.add_argument("--element", default=None, help="Optional PDB element symbol override")
    return parser

if __name__ == '__main__':
    import argparse
    args = add_args(argparse.ArgumentParser()).parse_args()
    main(args)


