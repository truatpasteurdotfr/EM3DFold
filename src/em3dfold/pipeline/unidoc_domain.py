import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Bio.PDB import MMCIFParser, PDBParser

from em3dfold.polymer_utils import residue_constants as rc


ALPHA = 0.43
CONTACT_CUTOFF = 8.0
Fragment = np.ndarray


@dataclass
class ResidueRecord:
    resseq: int
    icode: str
    resname: str
    chain_id: str
    coord_cb_or_ca: np.ndarray
    n: Optional[np.ndarray]
    ca: Optional[np.ndarray]
    c: Optional[np.ndarray]
    o: Optional[np.ndarray]


def _log_stage(message: str):
    print(message, flush=True)


def _load_structure(structure_path: str):
    suffix = Path(structure_path).suffix.lower()
    if suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    elif suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure format: {structure_path}")
    return parser.get_structure("structure", structure_path)


def _is_protein_residue(residue) -> bool:
    return residue.resname in rc.restype_3to1 and rc.restype3_is_prot(residue.resname)


def _extract_chain_records(structure_path: str) -> Dict[str, List[ResidueRecord]]:
    structure = _load_structure(structure_path)
    model = next(structure.get_models())

    chain_to_records: Dict[str, List[ResidueRecord]] = {}
    for chain in model:
        records: List[ResidueRecord] = []
        for residue in chain:
            if not _is_protein_residue(residue):
                continue

            atoms = {atom.name: atom.coord.astype(np.float32) for atom in residue}
            ca = atoms.get("CA")
            cb = atoms.get("CB")
            if ca is None and cb is None:
                continue

            coord_cb_or_ca = cb if cb is not None else ca
            hetfield, resseq, icode = residue.get_id()
            if hetfield != " ":
                continue

            records.append(
                ResidueRecord(
                    resseq=int(resseq),
                    icode=str(icode).strip(),
                    resname=residue.resname,
                    chain_id=chain.id,
                    coord_cb_or_ca=np.asarray(coord_cb_or_ca, dtype=np.float32),
                    n=None if atoms.get("N") is None else np.asarray(atoms["N"], dtype=np.float32),
                    ca=None if ca is None else np.asarray(ca, dtype=np.float32),
                    c=None if atoms.get("C") is None else np.asarray(atoms["C"], dtype=np.float32),
                    o=None if atoms.get("O") is None else np.asarray(atoms["O"], dtype=np.float32),
                )
            )
        if records:
            chain_to_records[chain.id] = records
    return chain_to_records


def _compute_contact_matrix(coords: np.ndarray, cutoff: float = CONTACT_CUTOFF) -> np.ndarray:
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return (dists < cutoff).astype(np.float32)


def _assign_secondary_structure(records: Sequence[ResidueRecord]) -> np.ndarray:
    try:
        import pydssp
    except ImportError as exc:
        raise ImportError(
            "pydssp is required for UniDoc secondary-structure parsing. "
            "Install it with `pip install pydssp`."
        ) from exc

    coords = np.zeros((len(records), 4, 3), dtype=np.float32)
    valid = np.zeros((len(records),), dtype=bool)
    for i, record in enumerate(records):
        if record.n is None or record.ca is None or record.c is None or record.o is None:
            continue
        coords[i, 0] = record.n
        coords[i, 1] = record.ca
        coords[i, 2] = record.c
        coords[i, 3] = record.o
        valid[i] = True

    ss = np.full((len(records),), "-", dtype="<U1")
    if np.any(valid):
        assigned = pydssp.assign(coords[valid], out_type="c3")
        assigned = np.asarray(assigned)
        if assigned.ndim == 2:
            assigned = assigned[0]
        ss[valid] = assigned.astype("<U1")

    # Match UniDoc's structure classes:
    # 1 -> coil/turn, 2 -> beta-like, 3 -> helix-like
    vsec = np.ones((len(records),), dtype=np.int32)
    vsec[ss == "E"] = 2
    vsec[ss == "H"] = 3
    return vsec


def _normalize_fragment(fragment: Sequence[int]) -> Fragment:
    fragment = np.asarray(fragment, dtype=np.int32)
    if fragment.size == 0:
        return fragment
    return np.unique(fragment)


def _ranges_from_missing(missing: Sequence[int], start: int, end_exclusive: int) -> List[Tuple[int, int]]:
    missing = np.asarray(missing, dtype=np.int32)
    if missing.size == 0:
        return [(start, end_exclusive - 1)]

    ranges: List[Tuple[int, int]] = []
    left = start
    for miss in missing.tolist():
        right = int(miss) - 1
        if left <= right:
            ranges.append((left, right))
        left = int(miss) + 1
    if left <= end_exclusive - 1:
        ranges.append((left, end_exclusive - 1))
    return ranges


def _ranges_to_string(ranges: Sequence[Tuple[int, int]]) -> str:
    return ",".join(f"{left}~{right}" for left, right in ranges)


def _fragment_to_domain_string(fragment: Sequence[int], residue_numbers: Sequence[int]) -> str:
    fragment = _normalize_fragment(fragment)
    if fragment.size == 0:
        return ""

    residue_numbers = np.asarray(residue_numbers, dtype=np.int32)
    covered = np.unique(residue_numbers[fragment])
    full_range = np.arange(int(covered[0]), int(covered[-1]) + 1, dtype=np.int32)
    missing = full_range[~np.isin(full_range, covered)]
    return _ranges_to_string(
        _ranges_from_missing(missing, int(covered[0]), int(covered[-1]) + 1)
    )


def _fragment_key(fragment: Sequence[int]) -> Tuple[int, ...]:
    fragment = _normalize_fragment(fragment)
    return tuple(int(x) for x in fragment.tolist())


def _ave_density(contact_matrix: np.ndarray, w: Sequence[int]) -> float:
    if len(w) == 0:
        return 0.0
    w = np.asarray(w, dtype=np.int32)
    sub = contact_matrix[np.ix_(w, w)]
    valid_mask = np.tril(np.ones_like(sub, dtype=bool), k=-3)
    density = float(sub[valid_mask].sum())
    return density / float(len(w))


def _inter_nnc(contact_matrix: np.ndarray, x1: Sequence[int], x2: Sequence[int]) -> float:
    if len(x1) == 0 or len(x2) == 0:
        return 0.0
    nc = float(contact_matrix[np.ix_(x1, x2)].sum())
    return nc / (pow(len(x1), ALPHA) * pow(len(x2), ALPHA))


def _build_prefix_sum(matrix: np.ndarray) -> np.ndarray:
    return np.pad(matrix.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")


def _rect_sum(prefix: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    if r0 >= r1 or c0 >= c1:
        return 0.0
    return float(prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0])


def _cut_domain(
    contact_matrix: np.ndarray,
    w: Sequence[int],
    n: Sequence[int],
    v: Optional[Sequence[int]] = None,
    use_secondary: bool = False,
    aggressive: bool = False,
) -> List[Fragment]:
    w = np.asarray(w, dtype=np.int32)
    n = np.asarray(n, dtype=np.int32)
    if v is not None:
        v = np.asarray(v, dtype=np.int32)
    length = len(w)
    if length == 0:
        return []

    sub_contact = contact_matrix[np.ix_(w, w)]
    prefix = _build_prefix_sum(sub_contact)
    threshold = _ave_density(contact_matrix, w) / 2.0
    min_nnc_1 = float("inf")
    min_nnc_2 = float("inf")
    cut1_idx = None
    cut21_idx = None
    cut22_idx = None

    if aggressive:
        one_cut_range = range(1, length)
        two_cut_k1_range = range(1, length)
        two_cut_delta = 1
    else:
        one_cut_range = range(30, length - 30) if length > 60 else range(0, 0)
        two_cut_k1_range = range(15, length - 15) if length > 60 else range(0, 0)
        two_cut_delta = 35

    for k in one_cut_range:
        if k <= 0 or k >= length:
            continue
        if use_secondary and v is not None and v[k] != 1:
            continue
        nc = _rect_sum(prefix, 0, k, k, length)
        denom = pow(k, ALPHA) * pow(length - k, ALPHA)
        if denom <= 0:
            continue
        nnc = nc / denom
        if nnc < min_nnc_1:
            min_nnc_1 = nnc
            cut1_idx = k

    for k1 in two_cut_k1_range:
        if use_secondary and v is not None and v[k1] != 1:
            continue
        start_k2 = k1 + two_cut_delta
        stop_k2 = (length - 15) if not aggressive else length
        for k2 in range(start_k2, stop_k2):
            if k2 >= length:
                break
            if use_secondary and v is not None and v[k2] != 1:
                continue
            if contact_matrix[w[k1], w[k2]] <= 0.5:
                continue

            middle_len = k2 - k1 - 1
            other_len = length - k2 + k1 + 1
            if middle_len <= 0 or other_len <= 0:
                continue

            nc = _rect_sum(prefix, k1 + 1, k2, 0, k1) + _rect_sum(prefix, k1 + 1, k2, k2, length)
            denom = pow(middle_len, ALPHA) * pow(other_len, ALPHA)
            nnc = nc / denom
            if nnc < min_nnc_2:
                min_nnc_2 = nnc
                cut21_idx = k1
                cut22_idx = k2

    compare_value = min(min_nnc_1, min_nnc_2)
    can_split = compare_value <= threshold if use_secondary else compare_value < threshold
    if can_split:
        if min_nnc_1 < min_nnc_2 and cut1_idx is not None:
            return [_normalize_fragment(w[:cut1_idx]), _normalize_fragment(w[cut1_idx:])]
        if cut21_idx is not None and cut22_idx is not None:
            middle = _normalize_fragment(w[cut21_idx + 1 : cut22_idx])
            outer = _normalize_fragment(np.concatenate([w[: cut21_idx + 1], w[cut22_idx:]]))
            return [middle, outer]

    return [_normalize_fragment(w)]


def _recursive_split(
    contact_matrix: np.ndarray,
    residue_numbers: Sequence[int],
    use_secondary: bool,
    vsec: Optional[Sequence[int]] = None,
    aggressive: bool = False,
) -> List[Fragment]:
    base_w = np.arange(len(residue_numbers), dtype=np.int32)
    base_n = np.asarray(residue_numbers, dtype=np.int32)
    base_vsec = None if vsec is None else np.asarray(vsec, dtype=np.int32)
    fragments = _cut_domain(
        contact_matrix,
        base_w,
        base_n,
        v=base_vsec,
        use_secondary=use_secondary,
        aggressive=aggressive,
    )

    while True:
        changed = False
        next_frags: List[Fragment] = []
        for frag in fragments:
            frag = _normalize_fragment(frag)
            w1 = frag
            n1 = base_n[w1]
            v1 = None if base_vsec is None else base_vsec[w1]
            split = _cut_domain(
                contact_matrix,
                w1,
                n1,
                v=v1,
                use_secondary=use_secondary,
                aggressive=aggressive,
            )
            if len(split) > 1:
                next_frags.extend(split)
                changed = True
            else:
                next_frags.append(split[0])
        fragments = next_frags
        if not changed:
            break
    return fragments


def _merge_domains(
    initial_fragments: Sequence[Fragment],
    contact_matrix: np.ndarray,
    residue_numbers: Sequence[int],
    max_iterations: int = 256,
    max_fragments_for_merge: int = 128,
    max_pair_checks_per_iteration: Optional[int] = None,
    log_label: Optional[str] = None,
) -> List[Fragment]:
    del residue_numbers
    fragments = [_normalize_fragment(frag) for frag in initial_fragments if len(frag) > 0]
    if len(fragments) > max_fragments_for_merge:
        _log_stage(
            "skip merge because fragment count {} exceeds threshold {}".format(
                len(fragments),
                max_fragments_for_merge,
            )
        )
        return fragments

    cache_density: Dict[Tuple[int, ...], float] = {}

    for iteration in range(max_iterations):
        pair_count = len(fragments) * (len(fragments) - 1) // 2
        if log_label is not None:
            _log_stage(
                f"{log_label}: merge iteration={iteration} fragment_count={len(fragments)} pair_count={pair_count}"
            )
        if (
            max_pair_checks_per_iteration is not None
            and pair_count > max_pair_checks_per_iteration
        ):
            _log_stage(
                "{}: stop merge because pair_count {} exceeds threshold {}".format(
                    log_label or "merge",
                    pair_count,
                    max_pair_checks_per_iteration,
                )
            )
            return fragments

        combine_pair: Optional[Tuple[int, int]] = None
        max_nnc = 0.0
        for i in range(len(fragments)):
            frag_i = fragments[i]
            key_i = _fragment_key(frag_i)
            if frag_i.size == 0:
                continue
            if key_i not in cache_density:
                cache_density[key_i] = _ave_density(contact_matrix, frag_i) / 2.0
            w1nnc = cache_density[key_i]
            for j in range(i):
                frag_j = fragments[j]
                key_j = _fragment_key(frag_j)
                if frag_j.size == 0:
                    continue
                if key_j not in cache_density:
                    cache_density[key_j] = _ave_density(contact_matrix, frag_j) / 2.0
                w2nnc = cache_density[key_j]
                nnc = _inter_nnc(contact_matrix, frag_i, frag_j)
                min_nnc = min(w1nnc, w2nnc)
                if nnc >= min_nnc:
                    diff = nnc - min_nnc
                    if diff > max_nnc:
                        max_nnc = diff
                        combine_pair = (i, j)
        if combine_pair is None:
            return fragments

        i, j = combine_pair
        merged_fragment = _normalize_fragment(np.concatenate([fragments[i], fragments[j]]))
        merged = [merged_fragment]
        for frag_idx, frag in enumerate(fragments):
            if frag_idx not in {i, j}:
                merged.append(frag)
        fragments = merged

    _log_stage(
        "stop merge after reaching max_iterations={}".format(max_iterations)
    )
    return fragments


def _finalize_domains(domains_out: Sequence[Fragment], residue_numbers: Sequence[int]) -> List[str]:
    domains: List[str] = []
    for fragment in domains_out:
        domain_str = _fragment_to_domain_string(fragment, residue_numbers)
        if domain_str:
            domains.append(domain_str)
    return domains


def _finalize_fragments(fragments_out: Sequence[Fragment], residue_numbers: Sequence[int]) -> List[str]:
    fragments: List[str] = []
    for fragment in fragments_out:
        fragment_str = _fragment_to_domain_string(fragment, residue_numbers)
        if fragment_str:
            fragments.append(fragment_str)
    return fragments


def _domain_string(domains: Sequence[str]) -> str:
    return "/".join(domains)


def _parse_chain_domains(records: Sequence[ResidueRecord]) -> Dict[str, object]:
    start_time = time.perf_counter()
    chain_label = records[0].chain_id

    stage_start = time.perf_counter()
    _log_stage(f"chain {chain_label}: collect residue numbers and CB/CA proxy coordinates")
    residue_numbers = [record.resseq for record in records]
    coords = np.stack([record.coord_cb_or_ca for record in records], axis=0)
    prepare_elapsed = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _log_stage(f"chain {chain_label}: build 8A contact matrix")
    contact_matrix = _compute_contact_matrix(coords)
    contact_elapsed = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _log_stage(f"chain {chain_label}: assign secondary structure with pydssp")
    vsec = _assign_secondary_structure(records)
    ss_elapsed = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _log_stage(f"chain {chain_label}: split large_domain candidates")
    large_fragments = _recursive_split(
        contact_matrix,
        residue_numbers,
        use_secondary=True,
        vsec=vsec,
        aggressive=False,
    )
    _log_stage(
        f"chain {chain_label}: large_domain candidate_count={len(large_fragments)}"
    )
    large_split_elapsed = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _log_stage(f"chain {chain_label}: finalize fragments")
    fragment_strings = _finalize_fragments(large_fragments, residue_numbers)
    fragment_finalize_elapsed = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _log_stage(f"chain {chain_label}: merge and finalize large_domain")
    large_domains = _finalize_domains(
        _merge_domains(
            large_fragments,
            contact_matrix,
            residue_numbers,
            max_iterations=256,
            max_fragments_for_merge=256,
        ),
        residue_numbers,
    )
    large_finalize_elapsed = time.perf_counter() - stage_start

    elapsed = time.perf_counter() - start_time
    return {
        "residue_count": len(records),
        "fragment_list": fragment_strings,
        "fragment": _domain_string(fragment_strings),
        "small_domain_list": fragment_strings,
        "small_domain": _domain_string(fragment_strings),
        "large_domain_list": large_domains,
        "large_domain": _domain_string(large_domains),
        "elapsed_seconds": elapsed,
        "stage_seconds": {
            "prepare": prepare_elapsed,
            "contact_matrix": contact_elapsed,
            "secondary_structure": ss_elapsed,
            "large_split": large_split_elapsed,
            "fragment_finalize": fragment_finalize_elapsed,
            "large_finalize": large_finalize_elapsed,
        },
    }


def parse_unidoc_domains(
    structure_path: str,
    chain_id: Optional[str] = None,
    output_small_domain: bool = False,
) -> Dict[str, object]:
    del output_small_domain
    _log_stage("load structure and extract chain records")
    chain_to_records = _extract_chain_records(structure_path)
    if chain_id is not None:
        if chain_id not in chain_to_records:
            raise ValueError(f"Chain {chain_id!r} not found in {structure_path}")
        return {chain_id: _parse_chain_domains(chain_to_records[chain_id])}

    output = {}
    for chain, records in chain_to_records.items():
        output[chain] = _parse_chain_domains(records)
    return output


def get_args():
    parser = argparse.ArgumentParser(
        description="Python UniDoc-style domain parsing with pydssp secondary structure."
    )
    parser.add_argument("-i", "--input", required=True, help="Input PDB/mmCIF structure")
    parser.add_argument("-c", "--chain", default=None, help="Optional chain ID")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output json path. Defaults to <input>.unidoc_domains.json",
    )
    parser.add_argument(
        "--output-small-domain",
        action="store_true",
        help="Deprecated. small_domain now aliases fragment and is always included.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    start_time = time.perf_counter()
    _log_stage("start UniDoc-style domain parsing")
    result = parse_unidoc_domains(
        args.input,
        chain_id=args.chain,
        output_small_domain=args.output_small_domain,
    )
    total_elapsed = time.perf_counter() - start_time

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.input).with_suffix("")) + ".unidoc_domains.json"

    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    for chain_id, chain_result in result.items():
        print(
            "Chain {} elapsed_seconds={:.4f}".format(
                chain_id,
                float(chain_result.get("elapsed_seconds", 0.0)),
            )
        )
        stage_seconds = chain_result.get("stage_seconds", {})
        if stage_seconds:
            print(
                "Chain {} stages prepare={:.4f} contact_matrix={:.4f} secondary_structure={:.4f} large_split={:.4f} fragment_finalize={:.4f} large_finalize={:.4f}".format(
                    chain_id,
                    float(stage_seconds.get("prepare", 0.0)),
                    float(stage_seconds.get("contact_matrix", 0.0)),
                    float(stage_seconds.get("secondary_structure", 0.0)),
                    float(stage_seconds.get("large_split", 0.0)),
                    float(stage_seconds.get("fragment_finalize", 0.0)),
                    float(stage_seconds.get("large_finalize", 0.0)),
                )
            )
    print(f"Total elapsed_seconds={total_elapsed:.4f}")
    print(f"Saved UniDoc-style domains to {output_path}")


if __name__ == "__main__":
    main()
