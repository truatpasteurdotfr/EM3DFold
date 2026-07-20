from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


_BRACKET_PAIRS: List[Tuple[str, str]] = [
    ("(", ")"),
    ("<", ">"),
    ("[", "]"),
    ("{", "}"),
] + [(chr(ord("A") + i), chr(ord("a") + i)) for i in range(26)]

_OPEN_TO_CLOSE = {left: right for left, right in _BRACKET_PAIRS}
_CLOSE_TO_OPEN = {right: left for left, right in _BRACKET_PAIRS}


def _validate_pair(i: int, j: int, length: int):
    if not (0 <= i < length and 0 <= j < length):
        raise ValueError(f"Pair index out of range: ({i}, {j}) for length {length}")
    if i == j:
        raise ValueError(f"Self pair is not allowed: ({i}, {j})")


def _normalize_pair(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _pairs_cross(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    i, j = left
    k, l = right
    return (i < k < j < l) or (k < i < l < j)


def dot_bracket_to_pair_list(dot_bracket: str) -> List[Tuple[int, int]]:
    stacks = {left: [] for left, _ in _BRACKET_PAIRS}
    pairs: List[Tuple[int, int]] = []

    for idx, token in enumerate(dot_bracket):
        if token == ".":
            continue
        if token in _OPEN_TO_CLOSE:
            stacks[token].append(idx)
            continue
        if token in _CLOSE_TO_OPEN:
            left = _CLOSE_TO_OPEN[token]
            if not stacks[left]:
                raise ValueError(f"Unmatched closing token {token!r} at position {idx}")
            pairs.append((stacks[left].pop(), idx))
            continue
        raise ValueError(f"Unsupported dot-bracket token {token!r} at position {idx}")

    for token, stack in stacks.items():
        if stack:
            raise ValueError(f"Unmatched opening token {token!r} at positions {stack}")

    return sorted((_normalize_pair(i, j) for i, j in pairs), key=lambda item: (item[0], item[1]))


def dot_bracket_to_pair_matrix(dot_bracket: str) -> np.ndarray:
    length = len(dot_bracket)
    matrix = np.zeros((length, length), dtype=np.float32)
    for i, j in dot_bracket_to_pair_list(dot_bracket):
        matrix[i, j] = 1.0
        matrix[j, i] = 1.0
    return matrix


def pair_list_to_dot_bracket(
    pair_list: Sequence[Tuple[int, int]],
    length: int,
) -> str:
    normalized_pairs = [_normalize_pair(int(i), int(j)) for i, j in pair_list]
    for i, j in normalized_pairs:
        _validate_pair(i, j, length)

    used = set()
    for i, j in normalized_pairs:
        if i in used or j in used:
            raise ValueError("Each residue can appear in at most one pair in dot-bracket output")
        used.add(i)
        used.add(j)

    sorted_pairs = sorted(normalized_pairs, key=lambda item: (item[0], item[1]))
    bracket_assignments: List[int] = []
    pairs_per_bracket: List[List[Tuple[int, int]]] = []

    for pair in sorted_pairs:
        assigned = False
        for bracket_idx, existing_pairs in enumerate(pairs_per_bracket):
            if any(_pairs_cross(pair, existing_pair) for existing_pair in existing_pairs):
                continue
            existing_pairs.append(pair)
            bracket_assignments.append(bracket_idx)
            assigned = True
            break
        if assigned:
            continue

        bracket_idx = len(pairs_per_bracket)
        if bracket_idx >= len(_BRACKET_PAIRS):
            raise ValueError("Too many pseudoknot levels for available dot-bracket symbols")
        pairs_per_bracket.append([pair])
        bracket_assignments.append(bracket_idx)

    tokens = ["."] * length
    for (i, j), bracket_idx in zip(sorted_pairs, bracket_assignments):
        left_token, right_token = _BRACKET_PAIRS[bracket_idx]
        tokens[i] = left_token
        tokens[j] = right_token
    return "".join(tokens)


def pair_list_to_pair_matrix(pair_list: Sequence[Tuple[int, int]], length: int) -> np.ndarray:
    matrix = np.zeros((length, length), dtype=np.float32)
    for i, j in pair_list:
        i, j = _normalize_pair(int(i), int(j))
        _validate_pair(i, j, length)
        matrix[i, j] = 1.0
        matrix[j, i] = 1.0
    return matrix


def pair_matrix_to_pair_list(
    pair_matrix: np.ndarray,
    threshold: float = 0.5,
    enforce_one_partner: bool = True,
    allow_pseudoknots: bool = True,
    min_separation: int = 0,
) -> List[Tuple[int, int]]:
    pair_matrix = np.asarray(pair_matrix, dtype=np.float32)
    if pair_matrix.ndim != 2 or pair_matrix.shape[0] != pair_matrix.shape[1]:
        raise ValueError("pair_matrix must be square")

    length = pair_matrix.shape[0]
    sym_matrix = 0.5 * (pair_matrix + pair_matrix.T)
    candidate_indices = np.triu_indices(length, k=1)
    candidate_scores = sym_matrix[candidate_indices]

    candidates = []
    for i, j, score in zip(candidate_indices[0], candidate_indices[1], candidate_scores):
        if j - i <= min_separation:
            continue
        if score < threshold:
            continue
        candidates.append((float(score), int(i), int(j)))

    candidates.sort(key=lambda item: (item[0], item[2] - item[1]), reverse=True)

    selected: List[Tuple[int, int]] = []
    used = set()
    for _, i, j in candidates:
        if enforce_one_partner and (i in used or j in used):
            continue
        pair = (i, j)
        if (not allow_pseudoknots) and any(_pairs_cross(pair, existing) for existing in selected):
            continue
        selected.append(pair)
        if enforce_one_partner:
            used.add(i)
            used.add(j)
    return selected


def pair_matrix_to_dot_bracket(
    pair_matrix: np.ndarray,
    threshold: float = 0.5,
    enforce_one_partner: bool = True,
    allow_pseudoknots: bool = True,
    min_separation: int = 0,
) -> Tuple[str, List[Tuple[int, int]]]:
    pair_list = pair_matrix_to_pair_list(
        pair_matrix,
        threshold=threshold,
        enforce_one_partner=enforce_one_partner,
        allow_pseudoknots=allow_pseudoknots,
        min_separation=min_separation,
    )
    return pair_list_to_dot_bracket(pair_list, length=int(np.asarray(pair_matrix).shape[0])), pair_list


def write_dbn(
    output_path: str,
    dot_bracket: str,
    sequence: Optional[str] = None,
    name: Optional[str] = None,
):
    path = Path(output_path)
    header = name if name is not None else path.stem
    sequence = sequence if sequence is not None else ("N" * len(dot_bracket))
    if len(sequence) != len(dot_bracket):
        raise ValueError("sequence length must match dot-bracket length")
    path.write_text(f">{header}\n{sequence}\n{dot_bracket}\n", encoding="utf-8")


def read_dbn(input_path: str) -> Tuple[str, str, str]:
    lines = [line.strip() for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("DBN file must contain at least sequence and dot-bracket lines")

    if lines[0].startswith(">"):
        if len(lines) < 3:
            raise ValueError("FASTA-style DBN file must contain header, sequence, and dot-bracket")
        name = lines[0][1:].strip()
        sequence = lines[1]
        dot_bracket = lines[2]
    else:
        name = Path(input_path).stem
        sequence = lines[0]
        dot_bracket = lines[1]

    if len(sequence) != len(dot_bracket):
        raise ValueError("DBN sequence length and dot-bracket length do not match")
    return name, sequence, dot_bracket


def dbn_file_to_pair_matrix(input_path: str) -> Tuple[str, str, np.ndarray]:
    name, sequence, dot_bracket = read_dbn(input_path)
    return name, sequence, dot_bracket_to_pair_matrix(dot_bracket)
