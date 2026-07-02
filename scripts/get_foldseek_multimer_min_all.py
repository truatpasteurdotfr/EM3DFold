#!/usr/bin/env python3
"""Get per-query maximum column 3 values from a Foldseek multimer TSV."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INPUT = Path("/data_gu03/taoli/em3dfold/usalign/foldseek/out.multimer.tsv.new")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="For each query in column 1, report max($3)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input file path [default: {DEFAULT_INPUT}]",
    )
    args = parser.parse_args()

    stats: dict[str, float] = {}

    with args.input.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            query = fields[0]
            try:
                value3 = float(fields[2])
            except ValueError:
                continue

            if query not in stats or value3 > stats[query]:
                stats[query] = value3

    for query in sorted(stats):
        print(f"{query} {stats[query]:.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
