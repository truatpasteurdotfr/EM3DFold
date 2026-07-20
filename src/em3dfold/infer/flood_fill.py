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


def _safe_edge_probability(
    edge_existence_dict,
    idx_exist_to_original,
    src_idx,
    dst_idx,
    next_idx,
    default_value=0.1,
):
    if edge_existence_dict is None or idx_exist_to_original is None:
        return float(default_value)

    src_original = int(idx_exist_to_original[src_idx])
    dst_original = int(idx_exist_to_original[dst_idx])
    edge_info = edge_existence_dict.get(src_original, {}).get(dst_original)
    if edge_info is None:
        return float(default_value)
    if next_idx is None:
        if np.isscalar(edge_info):
            return float(edge_info)
        edge_array = np.asarray(edge_info, dtype=np.float32).reshape(-1)
        if edge_array.size == 0:
            return float(default_value)
        return float(np.max(edge_array))

    edge_array = np.asarray(edge_info, dtype=np.float32).reshape(-1)
    if next_idx >= edge_array.size:
        return float(default_value)
    return float(edge_array[next_idx])


def _log_gaussian_score(distance, expected_distance, sigma, eps=1e-6):
    sigma = max(float(sigma), eps)
    diff = float(distance) - float(expected_distance)
    return -0.5 * (diff / sigma) ** 2


def _collect_protein_beam_candidates(
    tail_idx,
    c_positions,
    n_positions,
    ca_positions,
    kdtree_n,
    kdtree_ca,
    used_mask,
    cn_radius,
    ca_rescue_radius,
    center_exclusion_distance_threshold,
):
    cn_candidates = np.asarray(
        kdtree_n.query_ball_point(c_positions[tail_idx], r=cn_radius, return_sorted=True),
        dtype=np.int32,
    )
    ca_candidates = np.asarray(
        kdtree_ca.query_ball_point(ca_positions[tail_idx], r=ca_rescue_radius, return_sorted=True),
        dtype=np.int32,
    )

    cn_candidate_set = set()
    ca_candidate_set = set()
    if cn_candidates.size > 0:
        cn_candidate_set = {int(idx) for idx in cn_candidates.tolist()}
    if ca_candidates.size > 0:
        ca_candidate_set = {int(idx) for idx in ca_candidates.tolist()}

    merged_candidates = []
    for candidate_idx in sorted(cn_candidate_set | ca_candidate_set):
        if candidate_idx == int(tail_idx):
            continue
        if bool(used_mask[candidate_idx]):
            continue

        ca_distance = float(
            np.linalg.norm(ca_positions[tail_idx] - ca_positions[candidate_idx])
        )
        if (
            center_exclusion_distance_threshold is not None
            and center_exclusion_distance_threshold > 0
            and ca_distance < center_exclusion_distance_threshold
        ):
            continue

        cn_distance = float(
            np.linalg.norm(c_positions[tail_idx] - n_positions[candidate_idx])
        )
        if candidate_idx in cn_candidate_set and candidate_idx in ca_candidate_set:
            provenance = "both"
        elif candidate_idx in cn_candidate_set:
            provenance = "cn_primary"
        else:
            provenance = "ca_rescue"

        merged_candidates.append(
            {
                "candidate_idx": int(candidate_idx),
                "cn_distance": cn_distance,
                "ca_distance": ca_distance,
                "provenance": provenance,
            }
        )

    merged_candidates.sort(
        key=lambda item: (
            item["provenance"] == "ca_rescue",
            abs(item["cn_distance"] - 1.35),
            abs(item["ca_distance"] - 3.8),
        )
    )
    return merged_candidates


def beam_trace_with_edge_dual_protein(
    atom_positions,
    b_factors=None,
    bond_distance_threshold=2.1,
    center_exclusion_distance_threshold=1.6,
    last_idx=2,
    next_idx=0,
    edge_existence_dict=None,
    idx_exist_to_original=None,
    beam_width=16,
    max_candidates=5000,
    max_path_length=None,
    ca_rescue_radius=4.8,
    edge_prob_floor=1e-4,
    cn_expected_distance=1.35,
    cn_sigma=0.35,
    ca_expected_distance=3.8,
    ca_sigma=0.65,
    confidence_weight=0.25,
    frozen_mask=None,
):
    atom_positions = np.asarray(atom_positions, dtype=np.float32)
    if atom_positions.ndim != 3:
        raise ValueError("atom_positions must be [N, num_atoms, 3]")

    num_nodes = int(atom_positions.shape[0])
    if b_factors is None:
        b_factors = np.ones((num_nodes,), dtype=np.float32)
    else:
        b_factors = np.asarray(b_factors, dtype=np.float32).reshape(-1)
        if b_factors.shape[0] != num_nodes:
            raise ValueError("b_factors length does not match atom_positions")

    if frozen_mask is None:
        frozen_mask = np.zeros((num_nodes,), dtype=bool)
    else:
        frozen_mask = np.asarray(frozen_mask, dtype=bool).reshape(-1)
        if frozen_mask.shape[0] != num_nodes:
            raise ValueError("frozen_mask length does not match atom_positions")

    if max_path_length is None:
        max_path_length = num_nodes
    max_path_length = max(int(max_path_length), 1)
    beam_width = max(int(beam_width), 1)
    max_candidates = max(int(max_candidates), 1)

    c_positions = atom_positions[:, last_idx]
    n_positions = atom_positions[:, next_idx]
    ca_positions = atom_positions[:, 1]
    kdtree_n = cKDTree(n_positions)
    kdtree_ca = cKDTree(ca_positions)

    sorted_seed_indices = np.argsort(b_factors)[::-1]
    seed_indices = [
        int(idx) for idx in sorted_seed_indices.tolist() if not bool(frozen_mask[int(idx)])
    ]

    completed_candidates = []

    for seed_idx in seed_indices:
        if len(completed_candidates) >= max_candidates:
            break

        initial_used_mask = frozen_mask.copy()
        initial_used_mask[seed_idx] = True
        initial_confidence = float(b_factors[seed_idx]) * float(confidence_weight)
        active_states = [
            {
                "path": [int(seed_idx)],
                "used_mask": initial_used_mask,
                "tail_idx": int(seed_idx),
                "score_total": initial_confidence,
                "score_edge": 0.0,
                "score_cn": 0.0,
                "score_ca": 0.0,
                "score_confidence": initial_confidence,
                "edge_sources": [],
                "edge_scores": [],
            }
        ]

        while len(active_states) > 0 and len(completed_candidates) < max_candidates:
            next_states = []
            for state in active_states:
                if len(state["path"]) >= max_path_length:
                    completed_candidates.append(state)
                    if len(completed_candidates) >= max_candidates:
                        break
                    continue

                tail_idx = int(state["tail_idx"])
                candidates = _collect_protein_beam_candidates(
                    tail_idx=tail_idx,
                    c_positions=c_positions,
                    n_positions=n_positions,
                    ca_positions=ca_positions,
                    kdtree_n=kdtree_n,
                    kdtree_ca=kdtree_ca,
                    used_mask=state["used_mask"],
                    cn_radius=bond_distance_threshold,
                    ca_rescue_radius=ca_rescue_radius,
                    center_exclusion_distance_threshold=center_exclusion_distance_threshold,
                )

                if len(candidates) == 0:
                    completed_candidates.append(state)
                    if len(completed_candidates) >= max_candidates:
                        break
                    continue

                for candidate in candidates:
                    candidate_idx = int(candidate["candidate_idx"])
                    edge_prob = _safe_edge_probability(
                        edge_existence_dict=edge_existence_dict,
                        idx_exist_to_original=idx_exist_to_original,
                        src_idx=tail_idx,
                        dst_idx=candidate_idx,
                        next_idx=0,
                    )
                    edge_prob = max(float(edge_prob), float(edge_prob_floor))
                    score_edge = float(np.log(edge_prob))
                    score_cn = _log_gaussian_score(
                        candidate["cn_distance"],
                        expected_distance=cn_expected_distance,
                        sigma=cn_sigma,
                    )
                    score_ca = _log_gaussian_score(
                        candidate["ca_distance"],
                        expected_distance=ca_expected_distance,
                        sigma=ca_sigma,
                    )
                    score_confidence = float(b_factors[candidate_idx]) * float(confidence_weight)
                    total_increment = score_edge + score_cn + score_ca + score_confidence

                    new_used_mask = state["used_mask"].copy()
                    new_used_mask[candidate_idx] = True
                    next_states.append(
                        {
                            "path": state["path"] + [candidate_idx],
                            "used_mask": new_used_mask,
                            "tail_idx": candidate_idx,
                            "score_total": float(state["score_total"] + total_increment),
                            "score_edge": float(state["score_edge"] + score_edge),
                            "score_cn": float(state["score_cn"] + score_cn),
                            "score_ca": float(state["score_ca"] + score_ca),
                            "score_confidence": float(
                                state["score_confidence"] + score_confidence
                            ),
                            "edge_sources": state["edge_sources"] + [candidate["provenance"]],
                            "edge_scores": state["edge_scores"]
                            + [
                                {
                                    "src_idx": tail_idx,
                                    "dst_idx": candidate_idx,
                                    "edge_probability": float(edge_prob),
                                    "cn_distance": float(candidate["cn_distance"]),
                                    "ca_distance": float(candidate["ca_distance"]),
                                    "provenance": candidate["provenance"],
                                    "score_edge": float(score_edge),
                                    "score_cn": float(score_cn),
                                    "score_ca": float(score_ca),
                                    "score_confidence": float(score_confidence),
                                }
                            ],
                        }
                    )

            if len(completed_candidates) >= max_candidates:
                break
            if len(next_states) == 0:
                break

            next_states.sort(
                key=lambda state: (state["score_total"], len(state["path"])),
                reverse=True,
            )
            active_states = next_states[:beam_width]

    final_candidates = []
    for candidate_id, candidate in enumerate(completed_candidates[:max_candidates]):
        path = [int(idx) for idx in candidate["path"]]
        edge_sources = [str(source) for source in candidate["edge_sources"]]
        num_rescue_edges = int(sum(source == "ca_rescue" for source in edge_sources))
        num_both_edges = int(sum(source == "both" for source in edge_sources))
        final_candidates.append(
            {
                "candidate_id": int(candidate_id),
                "node_indices": path,
                "path_length": int(len(path)),
                "tail_idx": int(path[-1]),
                "score_total": float(candidate["score_total"]),
                "score_breakdown": {
                    "edge": float(candidate["score_edge"]),
                    "cn": float(candidate["score_cn"]),
                    "ca": float(candidate["score_ca"]),
                    "confidence": float(candidate["score_confidence"]),
                },
                "edge_sources": edge_sources,
                "edge_scores": candidate["edge_scores"],
                "num_ca_rescue_edges": num_rescue_edges,
                "num_both_edges": num_both_edges,
            }
        )

    return {
        "candidates": final_candidates,
        "settings": {
            "beam_width": beam_width,
            "max_candidates": max_candidates,
            "max_path_length": int(max_path_length),
            "bond_distance_threshold": float(bond_distance_threshold),
            "center_exclusion_distance_threshold": float(center_exclusion_distance_threshold),
            "ca_rescue_radius": float(ca_rescue_radius),
            "cn_expected_distance": float(cn_expected_distance),
            "ca_expected_distance": float(ca_expected_distance),
            "cn_sigma": float(cn_sigma),
            "ca_sigma": float(ca_sigma),
            "confidence_weight": float(confidence_weight),
            "edge_prob_floor": float(edge_prob_floor),
            "num_frozen_nodes": int(np.sum(frozen_mask)),
            "num_available_nodes": int(num_nodes - np.sum(frozen_mask)),
        },
    }

