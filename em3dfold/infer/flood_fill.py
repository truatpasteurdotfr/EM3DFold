import torch
import numpy as np
from scipy.spatial import cKDTree

def flood_fill(
    atom_positions,
    b_factors=None,
    bond_distance_threshold=2.1,
    last_idx=2,
    next_idx=0,
):
    if b_factors is None:
        b_factors = np.ones( len(atom_positions), dtype=np.float32 )

    c_positions = atom_positions[:, last_idx]
    n_positions = atom_positions[:, next_idx]
    kdtree = cKDTree(c_positions)
    b_factors_copy = np.copy(b_factors)

    chains = []
    chain_ends = {}
    while np.any(b_factors_copy != -1):
        idx = np.argmax(b_factors_copy)

        # find all possible neighbors
        possible_indices = np.array(
            kdtree.query_ball_point(n_positions[idx], r=bond_distance_threshold)
        )

        # then sort by distances
        if len(possible_indices) > 0:
            distances = np.linalg.norm(n_positions[possible_indices] - n_positions[idx], axis=1)
            sorted_indices = np.argsort(distances)
            possible_indices = possible_indices[sorted_indices]

        got_chain = False
        if len(possible_indices) > 0:
            for possible_prev_residue in possible_indices:
                if possible_prev_residue == idx:
                    continue
                if possible_prev_residue in chain_ends:
                    chains[chain_ends[possible_prev_residue]].append(idx)
                    chain_ends[idx] = chain_ends[possible_prev_residue]
                    del chain_ends[possible_prev_residue]
                    got_chain = True
                    break
                elif b_factors_copy[possible_prev_residue] >= 0.0:
                    chains.append([possible_prev_residue, idx])
                    chain_ends[idx] = len(chains) - 1
                    b_factors_copy[possible_prev_residue] = -1
                    got_chain = True
                    break

        if not got_chain:
            chains.append([idx])
            chain_ends[idx] = len(chains) - 1

        b_factors_copy[idx] = -1

    og_chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
    og_chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    n_chain_starts = n_positions[chain_starts]
    c_chain_ends = c_positions[chain_ends]
    N = len(chain_starts)
    spent_starts, spent_ends = set(), set()

    kdtree = cKDTree(n_chain_starts)

    no_improvement = 0
    chain_end_match = 0

    while no_improvement < 2 * N:
        found_match = False
        if chain_end_match in spent_ends:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N
            continue

        start_matches = kdtree.query_ball_point(
            c_chain_ends[chain_end_match], r=bond_distance_threshold, return_sorted=True
        )
        for chain_start_match in start_matches:
            if (
                chain_start_match not in spent_starts
                and chain_end_match != chain_start_match
            ):
                chain_start_match_reidx = np.nonzero(
                    chain_starts == og_chain_starts[chain_start_match]
                )[0][0]
                chain_end_match_reidx = np.nonzero(
                    chain_ends == og_chain_ends[chain_end_match]
                )[0][0]
                if chain_start_match_reidx == chain_end_match_reidx:
                    continue

                new_chain = (
                    chains[chain_end_match_reidx] + chains[chain_start_match_reidx]
                )

                chain_arange = np.arange(len(chains))
                tmp_chains = np.array(chains, dtype=object)[
                    chain_arange[
                        (chain_arange != chain_start_match_reidx)
                        & (chain_arange != chain_end_match_reidx)
                    ]
                ].tolist()
                tmp_chains.append(new_chain)
                chains = tmp_chains

                chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
                chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

                spent_starts.add(chain_start_match)
                spent_ends.add(chain_end_match)
                no_improvement = 0
                found_match = True
                chain_end_match = (chain_end_match + 1) % N
                break

        if not found_match:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N

    return chains

def edge_correction(
    possible_indices,
    idx,
    dmat,
    edge_existence_dict,
    idx_exist_to_original,
    next_idx=None,
    eps=1e-6,
):
    assert idx is not None

    # get connection probs
    existence = []
    idx_original = idx_exist_to_original[idx]

    for pidx in possible_indices:
        pidx_original = idx_exist_to_original[pidx]

        if pidx_original in edge_existence_dict[idx_original]:
            existence.append( edge_existence_dict[idx_original][pidx_original][next_idx] )
        else:
            existence.append( 0.1 )

    existence = np.array(existence, dtype=np.float32).flatten()
    connection_probs = np.exp(-4.0 * (dmat - 1.35 + eps) ** 2)

    return np.argsort(1 - connection_probs * existence)


def flood_fill_with_edge(
    atom_positions,
    b_factors=None,
    bond_distance_threshold=2.1,
    center_exclusion_distance_threshold=3.1,
    last_idx=2,
    next_idx=0,
    edge_existence_dict=None,
    idx_exist_to_original=None,
):
    if b_factors is None:
        b_factors = np.ones( len(atom_positions), dtype=np.float32 )

    center_positions = atom_positions[:, 1]

    c_positions = atom_positions[:, last_idx]
    n_positions = atom_positions[:, next_idx]
    kdtree = cKDTree(c_positions)
    b_factors_copy = np.copy(b_factors)

    chains = []
    chain_ends = {}
    while np.any(b_factors_copy != -1):
        idx = np.argmax(b_factors_copy)

        # find all possible neighbors
        possible_indices = np.array(
            kdtree.query_ball_point(n_positions[idx], r=bond_distance_threshold, return_sorted=True)
        )
        possible_indices = possible_indices.astype(np.int32)
        possible_indices = possible_indices[possible_indices != idx]

        # Exclude candidates whose center atom is too close to current residue center.
        if len(possible_indices) > 0 and center_exclusion_distance_threshold is not None and center_exclusion_distance_threshold > 0:
            center_dmat = np.sqrt(
                np.sum(
                    np.square(center_positions[idx][None] - center_positions[possible_indices]),
                    axis=-1,
                )
            )
            possible_indices = possible_indices[
                center_dmat >= center_exclusion_distance_threshold
            ]

        # edge correction
        if len(possible_indices) > 0:
            dmat = np.sqrt(np.sum(np.square(n_positions[idx][None] - c_positions[possible_indices]), axis=-1))
            possible_indices = possible_indices[
                edge_correction(
                    possible_indices,
                    idx,
                    dmat,
                    edge_existence_dict,
                    idx_exist_to_original,
                )
            ]

        got_chain = False
        if len(possible_indices) > 0:
            for possible_prev_residue in possible_indices:
                if possible_prev_residue == idx:
                    continue
                if possible_prev_residue in chain_ends:
                    chains[chain_ends[possible_prev_residue]].append(idx)
                    chain_ends[idx] = chain_ends[possible_prev_residue]
                    del chain_ends[possible_prev_residue]
                    got_chain = True
                    break
                elif b_factors_copy[possible_prev_residue] >= 0.0:
                    chains.append([possible_prev_residue, idx])
                    chain_ends[idx] = len(chains) - 1
                    b_factors_copy[possible_prev_residue] = -1
                    got_chain = True
                    break

        if not got_chain:
            chains.append([idx])
            chain_ends[idx] = len(chains) - 1

        b_factors_copy[idx] = -1

    og_chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
    og_chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    n_chain_starts = n_positions[chain_starts]
    c_chain_ends = c_positions[chain_ends]
    N = len(chain_starts)
    spent_starts, spent_ends = set(), set()

    kdtree = cKDTree(n_chain_starts)

    no_improvement = 0
    chain_end_match = 0

    while no_improvement < 2 * N:
        found_match = False
        if chain_end_match in spent_ends:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N
            continue

        start_matches = kdtree.query_ball_point(
            c_chain_ends[chain_end_match], r=bond_distance_threshold, return_sorted=True
        )

        if len(start_matches) > 0:
            start_matches = np.array(start_matches).astype(np.int32)
            dmat = np.sqrt(np.sum(np.square(c_chain_ends[chain_end_match][None] - n_chain_starts[start_matches]), axis=-1))
            start_matches = start_matches[
                edge_correction(
                    og_chain_starts[start_matches],
                    og_chain_ends[chain_end_match],
                    dmat,
                    edge_existence_dict,
                    idx_exist_to_original,
                )
            ]


        for chain_start_match in start_matches:
            if (
                chain_start_match not in spent_starts
                and chain_end_match != chain_start_match
            ):
                chain_start_match_reidx = np.nonzero(
                    chain_starts == og_chain_starts[chain_start_match]
                )[0][0]
                chain_end_match_reidx = np.nonzero(
                    chain_ends == og_chain_ends[chain_end_match]
                )[0][0]
                if chain_start_match_reidx == chain_end_match_reidx:
                    continue

                new_chain = (
                    chains[chain_end_match_reidx] + chains[chain_start_match_reidx]
                )

                chain_arange = np.arange(len(chains))
                tmp_chains = np.array(chains, dtype=object)[
                    chain_arange[
                        (chain_arange != chain_start_match_reidx)
                        & (chain_arange != chain_end_match_reidx)
                    ]
                ].tolist()
                tmp_chains.append(new_chain)
                chains = tmp_chains

                chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
                chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

                spent_starts.add(chain_start_match)
                spent_ends.add(chain_end_match)
                no_improvement = 0
                found_match = True
                chain_end_match = (chain_end_match + 1) % N
                break

        if not found_match:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N

    return chains

def flood_fill_with_edge_dual(
    atom_positions,
    b_factors=None,
    bond_distance_threshold=2.1,
    center_exclusion_distance_threshold=1.6,
    last_idx=2,
    next_idx=0,
    edge_existence_dict=None,
    idx_exist_to_original=None,
):
    if b_factors is None:
        b_factors = np.ones( len(atom_positions), dtype=np.float32 )

    center_positions = atom_positions[:, 1]

    c_positions = atom_positions[:, last_idx]
    n_positions = atom_positions[:, next_idx]
    kdtree = cKDTree(c_positions)
    b_factors_copy = np.copy(b_factors)

    chains = []
    chain_ends = {}
    while np.any(b_factors_copy != -1):
        idx = np.argmax(b_factors_copy)

        # find all possible neighbors
        possible_indices = np.array(
            kdtree.query_ball_point(n_positions[idx], r=bond_distance_threshold, return_sorted=True)
        )
        possible_indices = possible_indices.astype(np.int32)
        possible_indices = possible_indices[possible_indices != idx]

        # Exclude candidates whose center atom is too close to current residue center.
        if len(possible_indices) > 0 and center_exclusion_distance_threshold is not None and center_exclusion_distance_threshold > 0:
            center_dmat = np.sqrt(
                np.sum(
                    np.square(center_positions[idx][None] - center_positions[possible_indices]),
                    axis=-1,
                )
            )
            possible_indices = possible_indices[
                center_dmat >= center_exclusion_distance_threshold
            ]

        # edge correction
        if len(possible_indices) > 0:
            dmat = np.sqrt(np.sum(np.square(n_positions[idx][None] - c_positions[possible_indices]), axis=-1))
            possible_indices = possible_indices[
                edge_correction(
                    possible_indices,
                    idx,
                    dmat,
                    edge_existence_dict,
                    idx_exist_to_original,
                    next_idx=1,
                )
            ]

        got_chain = False
        if len(possible_indices) > 0:
            for possible_prev_residue in possible_indices:
                if possible_prev_residue == idx:
                    continue
                if possible_prev_residue in chain_ends:
                    chains[chain_ends[possible_prev_residue]].append(idx)
                    chain_ends[idx] = chain_ends[possible_prev_residue]
                    del chain_ends[possible_prev_residue]
                    got_chain = True
                    break
                elif b_factors_copy[possible_prev_residue] >= 0.0:
                    chains.append([possible_prev_residue, idx])
                    chain_ends[idx] = len(chains) - 1
                    b_factors_copy[possible_prev_residue] = -1
                    got_chain = True
                    break

        if not got_chain:
            chains.append([idx])
            chain_ends[idx] = len(chains) - 1

        b_factors_copy[idx] = -1

    og_chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
    og_chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    n_chain_starts = n_positions[chain_starts]
    c_chain_ends = c_positions[chain_ends]
    N = len(chain_starts)
    spent_starts, spent_ends = set(), set()

    kdtree = cKDTree(n_chain_starts)

    no_improvement = 0
    chain_end_match = 0

    while no_improvement < 2 * N:
        found_match = False
        if chain_end_match in spent_ends:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N
            continue

        start_matches = kdtree.query_ball_point(
            c_chain_ends[chain_end_match], r=bond_distance_threshold, return_sorted=True
        )

        if len(start_matches) > 0:
            start_matches = np.array(start_matches).astype(np.int32)
            dmat = np.sqrt(np.sum(np.square(c_chain_ends[chain_end_match][None] - n_chain_starts[start_matches]), axis=-1))

            # Edge connection with edge prob
            start_matches = start_matches[
                edge_correction(
                    og_chain_starts[start_matches],
                    og_chain_ends[chain_end_match],
                    dmat,
                    edge_existence_dict,
                    idx_exist_to_original,
                    next_idx=0,
                )
            ]


        for chain_start_match in start_matches:
            if (
                chain_start_match not in spent_starts
                and chain_end_match != chain_start_match
            ):
                chain_start_match_reidx = np.nonzero(
                    chain_starts == og_chain_starts[chain_start_match]
                )[0][0]
                chain_end_match_reidx = np.nonzero(
                    chain_ends == og_chain_ends[chain_end_match]
                )[0][0]
                if chain_start_match_reidx == chain_end_match_reidx:
                    continue

                new_chain = (
                    chains[chain_end_match_reidx] + chains[chain_start_match_reidx]
                )

                chain_arange = np.arange(len(chains))
                tmp_chains = np.array(chains, dtype=object)[
                    chain_arange[
                        (chain_arange != chain_start_match_reidx)
                        & (chain_arange != chain_end_match_reidx)
                    ]
                ].tolist()
                tmp_chains.append(new_chain)
                chains = tmp_chains

                chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
                chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

                spent_starts.add(chain_start_match)
                spent_ends.add(chain_end_match)
                no_improvement = 0
                found_match = True
                chain_end_match = (chain_end_match + 1) % N
                break

        if not found_match:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N

    return chains

