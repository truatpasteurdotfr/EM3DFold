#!/usr/bin/env python3
"""Extract the maximum USalign TM-score normalized by Structure_1."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


STRUCTURE_1_RE = re.compile(r"^Name of Structure_1:\s*(.+)$")
STRUCTURE_2_RE = re.compile(r"^Name of Structure_2:\s*(.+)$")
TM_SCORE_RE = re.compile(
    r"^TM-score=\s*([0-9]*\.?[0-9]+)\s+\(normalized by length of Structure_([12]):"
)


def parse_log(path: Path) -> tuple[float, str, str, int]:
    best_score = float("-inf")
    best_structure_1 = "-"
    best_structure_2 = "-"
    n_hits = 0

    current_structure_1 = "-"
    current_structure_2 = "-"

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            match = STRUCTURE_1_RE.match(line)
            if match:
                current_structure_1 = match.group(1).strip()
                continue

            match = STRUCTURE_2_RE.match(line)
            if match:
                current_structure_2 = match.group(1).strip()
                continue

            match = TM_SCORE_RE.match(line)
            if not match:
                continue

            score = float(match.group(1))
            normalized_by = match.group(2)
            if normalized_by != "1":
                continue

            n_hits += 1
            if score > best_score:
                best_score = score
                best_structure_1 = current_structure_1
                best_structure_2 = current_structure_2

    if n_hits == 0:
        raise ValueError(f"no Structure_1-normalized TM-score found in {path}")

    return best_score, best_structure_1, best_structure_2, n_hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the maximum TM-score normalized by Structure_1 from a USalign log."
    )
    parser.add_argument("log", type=Path, help="Path to a USalign .log file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra metadata instead of only the numeric max TM-score.",
    )
    args = parser.parse_args()

    score, structure_1, structure_2, n_hits = parse_log(args.log)
    if args.verbose:
        print(f"log\t{args.log}")
        print(f"structure_1\t{structure_1}")
        print(f"best_structure_2\t{structure_2}")
        print(f"max_tm_score_structure_1\t{score:.5f}")
        print(f"num_structure_1_scores\t{n_hits}")
    else:
        print(f"{score:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
