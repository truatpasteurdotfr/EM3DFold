#!/usr/bin/env python3
"""Get per-query max of min($3, $4) from a Foldseek multimer TSV."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INPUT = Path("/data_gu03/taoli/em3dfold/usalign/foldseek/out.multimer.tsv.new")


def read_query_order(list_path: Path | None) -> list[str] | None:
    if list_path is None:
        return None
    queries: list[str] = []
    with list_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.split()
            if fields:
                queries.append(fields[0])
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="For each query, compute max(min($3,$4)) across all targets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input file path [default: {DEFAULT_INPUT}]",
    )
    parser.add_argument(
        "--list",
        type=Path,
        default=None,
        help="Optional list file; if provided, only queries in list column 1 are reported, in list order.",
    )
    args = parser.parse_args()

    query_order = read_query_order(args.list)
    wanted = set(query_order) if query_order is not None else None
    best_per_query: dict[str, tuple[float, str]] = {}

    with args.input.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 4:
                continue
            query = fields[0]
            target = fields[1]
            if wanted is not None and query not in wanted:
                continue
            try:
                value3 = float(fields[2])
                value4 = float(fields[3])
            except ValueError:
                continue

            row_min = min(value3, value4)
            if query not in best_per_query or row_min > best_per_query[query][0]:
                best_per_query[query] = (row_min, target)

    if not best_per_query:
        raise SystemExit("no valid rows found")

    if query_order is not None:
        for query in query_order:
            if query in best_per_query:
                score, target = best_per_query[query]
                print(f"{query} {score:.5f} {target}")
            else:
                print(f"{query} - -")
    else:
        for query in sorted(best_per_query):
            score, target = best_per_query[query]
            print(f"{query} {score:.5f} {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
