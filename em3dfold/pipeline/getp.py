import os
import numpy as np
from scipy.spatial import cKDTree

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

def grid_to_points(
    grid, threshold, neighbour_distance_threshold, prune_distance=1.1,
):
    lattice = np.flip(get_lattice_meshgrid_np(grid.shape[-1], no_shift=True), -1)

    output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))

    points = lattice[grid > threshold, :].reshape(-1, 3)
    probs = grid[grid > threshold]

    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        n = 0

        new_points = np.copy(points)
        for p in points:
            neighbours = kdtree.query_ball_point(p, prune_distance)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx = np.argmax(probs[selection])
                prob_sum = np.sum(probs[selection])

                new_points[selection[keep_idx]] = (
                    np.sum(probs[selection][..., None] * points[selection], axis=0)
                    / prob_sum
                )
                probs[selection] = 0
                probs[selection[keep_idx]] = prob_sum

            n += 1

        points = new_points[probs > 0].reshape(-1, 3)
        probs = probs[probs > 0]

    kdtree = cKDTree(np.copy(points))
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            points[point_idx] = np.nan

    points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)

    output_points = points
    return output_points, output_points_before_pruning


def merge(coord, dens, d=1.0):
    print("# Merge distance = {:.4f}".format(d))

    n = len(coord)
    if n == 0:
        return

    # construct tree
    tree = cKDTree(coord)

    sorted_indices = np.argsort(dens)[::-1]
    kept = np.ones(n, dtype=bool)

    n_round = 0
    for i in sorted_indices:
        if not kept[i]:
            continue

        neighbors = tree.query_ball_point([coord[i]], r=d)[0]
        neighbors = np.asarray(neighbors).astype(np.int32)
        neighbors = neighbors[neighbors != i]

        kept[neighbors] = False

        n_round += 1

    return coord[kept], dens[kept]


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

    origin = np.asarray(origin, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    nxyz = np.asarray(grid.shape[::-1], dtype=np.int32)
    radius_grid = np.maximum(
        1,
        np.ceil(radius / np.maximum(voxel_size, 1e-6)).astype(np.int32),
    )
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

    order = np.argsort(candidate_dens)[::-1]
    selected_coords = []
    selected_dens = []
    selected_tree = None
    selected_points = None

    for idx in order:
        coord = candidate_coords[idx]
        dens = candidate_dens[idx]

        if selected_tree is not None:
            neighbours = selected_tree.query_ball_point(coord, r=supplement_merge_distance)
            if len(neighbours) > 0:
                continue

        selected_coords.append(coord)
        selected_dens.append(dens)

        selected_points = np.asarray(selected_coords, dtype=np.float32)
        selected_tree = cKDTree(selected_points)

        if max_points is not None and len(selected_coords) >= max_points:
            break

    if not selected_coords:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return (
        np.asarray(selected_coords, dtype=np.float32),
        np.asarray(selected_dens, dtype=np.float32),
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
    g2p_coords = np.zeros((0, 3), dtype=np.float32)
    g2p_dens = np.zeros((0,), dtype=np.float32)
    g2p_map = None
    g2p_origin = None
    g2p_vsize = None

    if args.run_g2p:
        # Run grid to points
        data, origin, nxyz, vsize = parse_map(args.map, False, None)
        data = enlarge_grid(data)

        maximum = np.percentile(data, 99.999)
        data = np.clip(data, 0.0, maximum)
        data = data / (data.max() + 1e-6)

        points, _ = grid_to_points(data, args.ratio, 6.0, prune_distance=1.5)
        g2p_coords = points + origin
        g2p_dens = sample_density_nearest(data, g2p_coords, origin, vsize)
        g2p_map = data
        g2p_origin = origin
        g2p_vsize = vsize
        print("# Done g2p n = {}".format(len(points)))

    # Run getp
    if args.run_getp:
        if len(g2p_coords) > 0:
            init_coords = g2p_coords
            init_pdb = None
        else:
            init_coords = None
            init_pdb = args.p

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

        # Merge again
        print("# Before merging n = {}".format(len(coords)))
        coords, dens = merge(coords, dens, args.dmerge)
        print("# After  merging n = {}".format(len(coords)))

        if args.run_g2p and args.fuse_g2p and len(g2p_coords) > 0:
            supplemental_coords, supplemental_dens = select_supplemental_g2p_points(
                getp_coords=coords,
                g2p_coords=g2p_coords,
                g2p_dens=g2p_dens,
                cover_distance=args.g2p_cover_distance,
                supplement_merge_distance=args.g2p_supplement_merge_distance,
                max_points=args.g2p_max_supplements,
            )
            print("# Candidate g2p supplements n = {}".format(len(supplemental_coords)))

            if len(supplemental_coords) > 0 and args.g2p_refine_radius > 0.0:
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

            if len(supplemental_coords) > 0:
                coords = np.concatenate([coords, supplemental_coords], axis=0)
                dens = np.concatenate([dens, supplemental_dens], axis=0)
                print("# After adding g2p supplements n = {}".format(len(coords)))
                coords, dens = merge(coords, dens, args.dmerge)
                print("# After final merge with supplements n = {}".format(len(coords)))

        final_dir = os.path.join(args.output, "merged.pdb")
        write_p(
            final_dir,
            coords, dens,
            atom_name=args.atom_name,
            res_name=args.res_name,
            chain_id=args.chain_id,
            element=args.element,
        )
        print("# Final coords write to {}".format(final_dir))

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
    parser.add_argument("--atom-name", default="CA", help="Atom name used when writing output PDB points")
    parser.add_argument("--res-name", default="GLY", help="Residue name used when writing output PDB points")
    parser.add_argument("--chain-id", default="A", help="Chain ID used when writing output PDB points")
    parser.add_argument("--element", default=None, help="Optional PDB element symbol override")
    return parser

if __name__ == '__main__':
    import argparse
    args = add_args(argparse.ArgumentParser()).parse_args()
    main(args)


