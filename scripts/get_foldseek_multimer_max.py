#!/usr/bin/env python3
"""Get max column 3 and 4 values for a given PDB id from a Foldseek multimer TSV."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INPUT = Path("/data_gu03/taoli/em3dfold/usalign/foldseek/out.multimer.tsv.new")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find max $3 and $4 for rows whose $1 matches the given pdbid."
    )
    parser.add_argument("pdbid", help="Query PDB id to match against column 1")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input file path [default: {DEFAULT_INPUT}]",
    )
    args = parser.parse_args()

    pdbid = args.pdbid.strip().upper()
    max_col3 = None
    max_col4 = None

    with args.input.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 4:
                continue
            if fields[0].upper() != pdbid:
                continue
            try:
                value3 = float(fields[2])
                value4 = float(fields[3])
            except ValueError:
                continue

            if max_col3 is None or value3 > max_col3:
                max_col3 = value3
            if max_col4 is None or value4 > max_col4:
                max_col4 = value4

    if max_col3 is None or max_col4 is None:
        raise SystemExit(f"no rows found for pdbid={pdbid} in {args.input}")

    print(f"{max_col3:.5f} {max_col4:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
