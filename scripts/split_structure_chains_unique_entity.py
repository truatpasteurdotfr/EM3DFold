#!/usr/bin/env python3
"""Split a structure into per-chain protein/RNA/DNA files, keeping one random chain per entity.

For mmCIF input, polymer entities are read from `_entity_poly`, and one chain is
randomly selected from each entity. For PDB input, entity information is
unavailable, so each chain is treated as a unique entity.

Output naming is:
- <prefix>.chain.<chain_id>.prot.<suffix>
- <prefix>.chain.<chain_id>.rna.<suffix>
- <prefix>.chain.<chain_id>.dna.<suffix>
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def _ensure_repo_paths_on_path() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    src_dir = repo_root / "src"
    for path in (script_dir, src_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_repo_paths_on_path()
import split_structure_chains as split_mod  # noqa: E402
from get_entity_uniprot_ids import _ensure_list, _normalize_token, _parse_mmcif_fields  # noqa: E402


PDB_SUFFIXES = {".pdb", ".ent"}
CIF_SUFFIXES = {".cif", ".mmcif"}


def _read_chain_ids(structure_path: Path) -> list[str]:
    return split_mod._collect_original_chain_ids(structure_path)


def _is_mmcif_path(path: Path) -> bool:
    return path.suffix.lower() in CIF_SUFFIXES


def _polymer_entity_to_chain_ids(mmcif: dict) -> dict[str, list[str]]:
    entity_ids = _ensure_list(mmcif.get("_entity_poly.entity_id"))
    strand_ids = _ensure_list(mmcif.get("_entity_poly.pdbx_strand_id"))

    mapping: dict[str, list[str]] = {}
    row_count = max(len(entity_ids), len(strand_ids))
    for idx in range(row_count):
        entity_id = _normalize_token(entity_ids[idx] if idx < len(entity_ids) else None)
        strand_text = _normalize_token(strand_ids[idx] if idx < len(strand_ids) else None)
        if entity_id is None or strand_text is None:
            continue
        chain_ids = [
            chain_id.strip()
            for chain_id in str(strand_text).split(",")
            if chain_id.strip() and chain_id.strip() not in {"?", "."}
        ]
        if chain_ids:
            mapping[entity_id] = chain_ids
    return mapping


def _select_unique_entity_chains(structure_path: Path, seed: int) -> tuple[set[str], list[tuple[str, str, list[str], str]]]:
    chain_ids = _read_chain_ids(structure_path)
    chain_id_set = set(chain_ids)
    rng = random.Random(seed)

    if not _is_mmcif_path(structure_path):
        selected = {chain_id for chain_id in chain_ids if chain_id}
        decisions = [
            (f"chain_{idx}", chain_id or f"chain{idx}", [chain_id or f"chain{idx}"], chain_id or f"chain{idx}")
            for idx, chain_id in enumerate(chain_ids)
        ]
        return selected, decisions

    mmcif = _parse_mmcif_fields(structure_path)
    entity_to_chains = _polymer_entity_to_chain_ids(mmcif)

    selected: set[str] = set()
    decisions: list[tuple[str, str, list[str], str]] = []
    all_entity_chains: set[str] = set()
    for entity_id in sorted(entity_to_chains, key=lambda value: (len(value), value)):
        candidates = [chain_id for chain_id in entity_to_chains[entity_id] if chain_id in chain_id_set]
        if not candidates:
            continue
        all_entity_chains.update(candidates)
        chosen = rng.choice(candidates)
        selected.add(chosen)
        decisions.append((entity_id, chosen, candidates, "entity"))

    for chain_idx, chain_id in enumerate(chain_ids):
        if not chain_id or chain_id in all_entity_chains:
            continue
        selected.add(chain_id)
        decisions.append((f"orphan_chain_{chain_idx}", chain_id, [chain_id], "orphan"))

    return selected, decisions


def split_structure_chains_unique_entity(
    structure_path: Path,
    output_dir: Path,
    *,
    prefix: str,
    suffix: str,
    seed: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_chain_ids, decisions = _select_unique_entity_chains(structure_path, seed)
    for entity_id, chosen, candidates, reason in decisions:
        print(
            f"select\tgroup={entity_id}\treason={reason}\tchosen={chosen}\t"
            f"candidates={','.join(candidates)}"
        )

    chain_ids = split_mod._collect_original_chain_ids(structure_path)
    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = split_mod.read_pdb(
        str(structure_path),
        keep_valid=False,
        return_bfactor=True,
    )
    if len(atom_pos) == 0:
        return []

    (
        chains_atom_pos,
        chains_atom_mask,
        chains_res_type,
        chains_res_idx,
        chains_bfactor,
    ) = split_mod.convert_to_chains(
        chain_idx,
        atom_pos,
        atom_mask,
        res_type,
        res_idx,
        bfactor,
    )

    written_paths: list[Path] = []
    n_chain = len(chains_atom_pos)
    for chain_local_idx in range(max(len(chain_ids), n_chain)):
        original_chain_id = chain_ids[chain_local_idx] if chain_local_idx < len(chain_ids) else ""
        chain_name = split_mod._sanitize_chain_id(original_chain_id, chain_local_idx)

        if chain_local_idx >= n_chain:
            print(f"skip\tchain={original_chain_id or chain_local_idx}\treason=no_supported_polymer_residues")
            continue
        if original_chain_id not in selected_chain_ids:
            print(f"skip\tchain={original_chain_id or chain_local_idx}\treason=not_selected_by_entity")
            continue

        chain_atom_pos = split_mod.np.asarray(chains_atom_pos[chain_local_idx])
        chain_atom_mask = split_mod.np.asarray(chains_atom_mask[chain_local_idx])
        chain_res_type = split_mod.np.asarray(chains_res_type[chain_local_idx])
        chain_res_idx = split_mod.np.asarray(chains_res_idx[chain_local_idx])
        chain_bfactor = split_mod.np.asarray(chains_bfactor[chain_local_idx])

        prot_mask = chain_res_type < 20
        dna_mask = (chain_res_type >= 20) & (chain_res_type < 24)
        rna_mask = (chain_res_type >= 24) & (chain_res_type < 28)
        prot_count = int(split_mod.np.count_nonzero(prot_mask))
        dna_count = int(split_mod.np.count_nonzero(dna_mask))
        rna_count = int(split_mod.np.count_nonzero(rna_mask))

        if prot_count + dna_count + rna_count <= 0:
            print(f"skip	chain={original_chain_id or chain_local_idx}	reason=no_supported_polymer_residues")
            continue

        writer_chain_index = split_mod._resolve_writer_chain_index(original_chain_id, chain_local_idx)

        if prot_count > 0:
            output_path = output_dir / f"{prefix}.chain.{chain_name}.prot.{suffix}"
            split_mod._write_chain_subset(
                output_path,
                atom_pos=chain_atom_pos[prot_mask],
                atom_mask=chain_atom_mask[prot_mask],
                res_type=chain_res_type[prot_mask],
                res_idx=chain_res_idx[prot_mask],
                bfactor=chain_bfactor[prot_mask],
                writer_chain_index=writer_chain_index,
                output_chain_id=original_chain_id or chain_name,
                suffix=suffix,
            )
            print(
                f"write	chain={original_chain_id or chain_local_idx}	kind=prot	"
                f"output={output_path}	residues={prot_count}"
            )
            written_paths.append(output_path)

        if rna_count > 0:
            output_path = output_dir / f"{prefix}.chain.{chain_name}.rna.{suffix}"
            split_mod._write_chain_subset(
                output_path,
                atom_pos=chain_atom_pos[rna_mask],
                atom_mask=chain_atom_mask[rna_mask],
                res_type=chain_res_type[rna_mask],
                res_idx=chain_res_idx[rna_mask],
                bfactor=chain_bfactor[rna_mask],
                writer_chain_index=writer_chain_index,
                output_chain_id=original_chain_id or chain_name,
                suffix=suffix,
            )
            print(
                f"write	chain={original_chain_id or chain_local_idx}	kind=rna	"
                f"output={output_path}	residues={rna_count}"
            )
            written_paths.append(output_path)

        if dna_count > 0:
            output_path = output_dir / f"{prefix}.chain.{chain_name}.dna.{suffix}"
            split_mod._write_chain_subset(
                output_path,
                atom_pos=chain_atom_pos[dna_mask],
                atom_mask=chain_atom_mask[dna_mask],
                res_type=chain_res_type[dna_mask],
                res_idx=chain_res_idx[dna_mask],
                bfactor=chain_bfactor[dna_mask],
                writer_chain_index=writer_chain_index,
                output_chain_id=original_chain_id or chain_name,
                suffix=suffix,
            )
            print(
                f"write	chain={original_chain_id or chain_local_idx}	kind=dna	"
                f"output={output_path}	residues={dna_count}"
            )
            written_paths.append(output_path)

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a structure into per-chain protein/RNA/DNA files, keeping one random chain per entity.",
    )
    parser.add_argument("structure", help="Input PDB/mmCIF file")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output file prefix. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--suffix",
        choices=("cif", "pdb"),
        default="cif",
        help="Output structure format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to choose one chain per entity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    structure_path = Path(args.structure).expanduser().resolve()
    if not structure_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    prefix = (args.prefix or structure_path.stem).strip()
    if not prefix:
        raise SystemExit("Output prefix must not be empty")

    written_paths = split_structure_chains_unique_entity(
        structure_path,
        output_dir,
        prefix=prefix,
        suffix=args.suffix,
        seed=int(args.seed),
    )
    if not written_paths:
        raise SystemExit(f"No supported polymer chains were written for {structure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
