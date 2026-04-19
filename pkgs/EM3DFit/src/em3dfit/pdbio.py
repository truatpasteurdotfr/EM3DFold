from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser

from em3dfit.score import euler_to_matrix
from em3dfit.types import Chain, DomainLink, PDBModel, SegmentLink, linear_segment_adjacency

MAINCHAIN_ATOMS = {" N  ", " CA ", " C  "}
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_SUFFIXES = {".pdb", ".ent"}
CIF_SUFFIXES = {".cif", ".mmcif"}


def _parse_xyz(line: str) -> np.ndarray:
    return np.asarray(
        [
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ],
        dtype=np.float32,
    )


def _normalize_output_text(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _read_structure_lines(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in PDB_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if suffix in CIF_SUFFIXES:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(path.stem or "model", str(path))
        buffer = StringIO()
        io = PDBIO()
        io.set_structure(structure)
        io.save(buffer, write_end=False)
        return buffer.getvalue().splitlines()
    raise ValueError(f"Unsupported structure format: {path}")


def _write_structure_lines(path: Path, lines: list[str]) -> None:
    suffix = path.suffix.lower()
    text = _normalize_output_text(lines)
    if suffix in PDB_SUFFIXES:
        path.write_text(text, encoding="utf-8")
        return
    if suffix in CIF_SUFFIXES:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(path.stem or "model", StringIO(text))
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(str(path))
        return
    raise ValueError(f"Unsupported structure format: {path}")


def _finalize_chain(
    chains: list[Chain],
    chain_index: int,
    main_coords: list[np.ndarray],
    residue_numbers: list[int],
    residue_id_labels: list[str],
    segment_numbers: list[int],
    is_ill: bool,
) -> None:
    if not main_coords:
        return

    coords = np.asarray(main_coords, dtype=np.float32)
    residues = np.asarray(residue_numbers, dtype=np.int32)
    segments = np.asarray(segment_numbers, dtype=np.int32)
    if is_ill or segments.size == 0 or np.any(segments <= 0):
        segments = np.ones_like(residues, dtype=np.int32)
        is_ill = True

    n_segments = int(segments.max(initial=1))
    frag_numbers = np.ones_like(segments, dtype=np.int32)
    frag = 1
    for idx in range(1, len(segments)):
        if segments[idx] != segments[idx - 1]:
            frag += 1
        frag_numbers[idx] = frag

    centroid = ((coords.min(axis=0) + coords.max(axis=0)) / 2.0).astype(np.float32, copy=False)
    chains.append(
        Chain(
            index=chain_index,
            n_residues=len(np.unique(residues)),
            n_atoms=len(coords),
            n_segments=n_segments,
            n_frags=int(frag_numbers.max(initial=1)),
            centroid=centroid,
            coords=coords,
            residue_numbers=residues,
            segment_numbers=segments,
            segment_adjacency=linear_segment_adjacency(n_segments),
            frag_numbers=frag_numbers,
            weights=np.ones(len(coords), dtype=np.float32),
            is_ill=is_ill,
            residue_id_labels=tuple(residue_id_labels),
        )
    )


def _parse_remark_link(line: str, chain_index: int) -> DomainLink:
    fields = line[11:].split()
    if len(fields) < 4:
        raise ValueError(f"Invalid REMARK LINK record for chain {chain_index}: {line.rstrip()}")
    return DomainLink(
        chain_index=chain_index,
        residue_id_a=fields[0].strip(),
        residue_id_b=fields[1].strip(),
        segment_a=int(fields[2]),
        segment_b=int(fields[3]),
    )


def _apply_remark_links(chains: list[Chain], links: list[DomainLink]) -> None:
    if not links:
        return
    links_by_chain: dict[int, list[DomainLink]] = {}
    for link in links:
        links_by_chain.setdefault(link.chain_index, []).append(link)

    for chain in chains:
        chain_links = links_by_chain.get(chain.index, [])
        if not chain_links or chain.is_ill or chain.n_segments <= 1:
            continue
        adjacency = np.zeros((chain.n_segments, chain.n_segments), dtype=np.bool_)
        segment_links: list[SegmentLink] = []
        labels = chain.residue_id_labels
        if labels is None:
            raise ValueError(f"Chain {chain.index} is missing residue labels needed for REMARK LINK resolution.")
        label_array = np.asarray(labels, dtype=object)
        for link in chain_links:
            left = int(link.segment_a)
            right = int(link.segment_b)
            if left == right:
                raise ValueError(f"Chain {chain.index} contains a self-link for segment {left}.")
            if left < 1 or right < 1 or left > chain.n_segments or right > chain.n_segments:
                raise ValueError(
                    f"Chain {chain.index} REMARK LINK references out-of-range segment(s): {left}, {right}."
            )
            adjacency[left - 1, right - 1] = True
            adjacency[right - 1, left - 1] = True
            left_ids = np.flatnonzero(
                (chain.segment_numbers == left) & (label_array == link.residue_id_a)
            )
            right_ids = np.flatnonzero(
                (chain.segment_numbers == right) & (label_array == link.residue_id_b)
            )
            if left_ids.size == 0:
                raise ValueError(
                    f"Chain {chain.index} REMARK LINK residue {link.residue_id_a} was not found in segment {left}."
                )
            if right_ids.size == 0:
                raise ValueError(
                    f"Chain {chain.index} REMARK LINK residue {link.residue_id_b} was not found in segment {right}."
                )
            segment_links.append(
                SegmentLink(
                    left_segment=left,
                    right_segment=right,
                    left_anchor_index=int(left_ids[left_ids.size // 2]),
                    right_anchor_index=int(right_ids[right_ids.size // 2]),
                    residue_id_a=link.residue_id_a,
                    residue_id_b=link.residue_id_b,
                )
            )

        visited = np.zeros(chain.n_segments, dtype=np.bool_)
        stack = [0]
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True
            neighbors = np.flatnonzero(adjacency[current])
            stack.extend(int(neighbor) for neighbor in neighbors if not visited[neighbor])
        if not bool(np.all(visited)):
            raise ValueError(f"Chain {chain.index} REMARK LINK graph contains disconnected segment islands.")
        chain.segment_adjacency = adjacency
        chain.segment_links = segment_links


def read_pdb(path: str | Path) -> PDBModel:
    path = Path(path)
    lines = _read_structure_lines(path)

    chains: list[Chain] = []
    links: list[DomainLink] = []
    main_coords: list[np.ndarray] = []
    residue_numbers: list[int] = []
    residue_id_labels: list[str] = []
    segment_numbers: list[int] = []
    chain_index = 1
    total_residues = 0
    current_is_ill = False
    active_chain = False
    previous_residue_field = ""

    for raw_line in lines:
        line = raw_line.rstrip("\n").ljust(80)
        record = line[:6]
        if record == "ATOM  ":
            active_chain = True
            residue_field = line[17:27]
            if residue_field != previous_residue_field:
                total_residues += 1
                previous_residue_field = residue_field

            atom_name = line[12:16]
            if atom_name not in MAINCHAIN_ATOMS:
                continue

            seg_text = line[70:74].strip()
            if seg_text and seg_text.isdigit() and int(seg_text) > 0 and not current_is_ill:
                segment_id = int(seg_text)
            else:
                segment_id = 1
                current_is_ill = True

            main_coords.append(_parse_xyz(line))
            residue_numbers.append(total_residues)
            residue_id_labels.append(line[22:27].strip())
            segment_numbers.append(segment_id)
        elif line.startswith("TER") and active_chain:
            _finalize_chain(
                chains,
                chain_index,
                main_coords,
                residue_numbers,
                residue_id_labels,
                segment_numbers,
                current_is_ill,
            )
            chain_index += 1
            main_coords = []
            residue_numbers = []
            residue_id_labels = []
            segment_numbers = []
            current_is_ill = False
            active_chain = False
            previous_residue_field = ""
        elif line.startswith("REMARK LINK"):
            links.append(_parse_remark_link(line, chain_index))

    if active_chain:
        _finalize_chain(
            chains,
            chain_index,
            main_coords,
            residue_numbers,
            residue_id_labels,
            segment_numbers,
            current_is_ill,
        )

    if not chains:
        raise ValueError(f"{path} does not contain any main-chain atoms.")

    _apply_remark_links(chains, links)
    return PDBModel(path=path, lines=lines, chains=chains, total_residues=total_residues, links=links)


def write_mcp_pdb(path: str | Path, ldps: np.ndarray, densities: np.ndarray) -> None:
    path = Path(path)
    lines: list[str] = []
    for idx, (coord, density) in enumerate(zip(ldps, densities, strict=True), start=1):
        lines.append(
            f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"{1.00:6.2f}{float(density):6.2f}          C"
        )
        lines.append("TER")
    _write_structure_lines(path, lines)


def write_scored_pdb(model: PDBModel, output_path: str | Path) -> None:
    output_path = Path(output_path)
    chains = model.chains

    n_segments = sum(chain.n_segments for chain in chains)
    n_frags = sum(chain.n_frags for chain in chains)
    n_residues = model.total_residues

    chain_scores = np.asarray(
        [float(np.mean(chain.scores)) if chain.scores is not None and len(chain.scores) else 0.0 for chain in chains],
        dtype=np.float32,
    )
    segment_scores = np.zeros(n_segments, dtype=np.float32)
    segment_denoms = np.zeros(n_segments, dtype=np.int32)
    frag_scores = np.zeros(n_frags, dtype=np.float32)
    frag_denoms = np.zeros(n_frags, dtype=np.int32)
    residue_scores = np.zeros(n_residues, dtype=np.float32)
    residue_denoms = np.zeros(n_residues, dtype=np.int32)

    segment_offset = 0
    frag_offset = 0
    for chain in chains:
        if chain.scores is None:
            continue
        for atom_idx, score in enumerate(chain.scores):
            seg = segment_offset + int(chain.segment_numbers[atom_idx]) - 1
            frag = frag_offset + int(chain.frag_numbers[atom_idx]) - 1
            residue = int(chain.residue_numbers[atom_idx]) - 1
            segment_scores[seg] += score
            segment_denoms[seg] += 1
            frag_scores[frag] += score
            frag_denoms[frag] += 1
            residue_scores[residue] += score
            residue_denoms[residue] += 1
        segment_offset += chain.n_segments
        frag_offset += chain.n_frags

    segment_denoms = np.maximum(segment_denoms, 1)
    frag_denoms = np.maximum(frag_denoms, 1)
    residue_denoms = np.maximum(residue_denoms, 1)
    segment_scores /= segment_denoms
    frag_scores /= frag_denoms
    residue_scores /= residue_denoms

    chain_scoresx = chain_scores.copy()
    segment_scoresx = np.zeros_like(segment_scores)
    frag_scoresx = np.zeros_like(frag_scores)
    residue_scoresx = np.zeros_like(residue_scores)

    segment_offset = 0
    frag_offset = 0
    for chain_idx, chain in enumerate(chains):
        seen_segments: set[int] = set()
        seen_frags: set[int] = set()
        seen_residues: set[int] = set()
        for atom_idx in range(chain.n_atoms):
            seg = segment_offset + int(chain.segment_numbers[atom_idx]) - 1
            frag = frag_offset + int(chain.frag_numbers[atom_idx]) - 1
            residue = int(chain.residue_numbers[atom_idx]) - 1
            if residue not in seen_residues:
                residue_scoresx[residue] = 0.7 * residue_scores[residue] + 0.3 * frag_scores[frag]
                seen_residues.add(residue)
            if frag not in seen_frags:
                frag_scoresx[frag] = 0.7 * frag_scores[frag] + 0.3 * segment_scores[seg]
                seen_frags.add(frag)
            if seg not in seen_segments:
                segment_scoresx[seg] = 0.7 * segment_scores[seg] + 0.3 * chain_scoresx[chain_idx]
                seen_segments.add(seg)
        segment_offset += chain.n_segments
        frag_offset += chain.n_frags

    output_lines: list[str] = []
    segment_offset = 0
    frag_offset = 0
    for chain_idx, chain in enumerate(chains):
        chain_label = ALPHABET[chain_idx % len(ALPHABET)]
        output_lines.append(f"REMARK chain score   {chain_label} {chain_scoresx[chain_idx]:8.3f}")
        for seg_idx in range(chain.n_segments):
            output_lines.append(
                f"REMARK domain score{segment_offset + seg_idx + 1:4d} {segment_scoresx[segment_offset + seg_idx]:8.3f}"
            )
        for frag_idx in range(chain.n_frags):
            output_lines.append(
                f"REMARK frag score {frag_offset + frag_idx + 1:4d} {frag_scoresx[frag_offset + frag_idx]:8.3f}"
            )
        segment_offset += chain.n_segments
        frag_offset += chain.n_frags

    nr = 0
    nc = 0
    na = 0
    segment_offset = 0
    frag_offset = 0
    previous_residue_field = ""
    current_frag = 1
    current_seg = 1
    for raw_line in model.lines:
        line = raw_line.rstrip("\n").ljust(80)
        record = line[:6]
        out = list(line[:80])
        if record == "ATOM  ":
            residue_field = line[17:27]
            if residue_field != previous_residue_field:
                nr += 1
                previous_residue_field = residue_field

            atom_name = line[12:16]
            if atom_name in MAINCHAIN_ATOMS and nc < len(chains):
                na += 1
                current_frag = frag_offset + int(chains[nc].frag_numbers[na - 1])
                current_seg = segment_offset + int(chains[nc].segment_numbers[na - 1])

            score_text = f"{residue_scoresx[max(nr - 1, 0)]:6.2f}"
            ids_text = f"{current_frag:4d}{current_seg:4d}"
            out[60:66] = list(score_text)
            out[66:74] = list(ids_text)
        elif line.startswith("TER"):
            if nc < len(chains):
                segment_offset += chains[nc].n_segments
                frag_offset += chains[nc].n_frags
            nc += 1
            na = 0
            previous_residue_field = ""
        output_lines.append("".join(out).rstrip())
    _write_structure_lines(output_path, output_lines)


def write_fitted_pdb(model: PDBModel, output_path: str | Path) -> None:
    output_path = Path(output_path)
    residue_maps: list[dict[int, int]] = []
    for chain in model.chains:
        residue_to_segment: dict[int, int] = {}
        for residue_number, segment_number in zip(chain.residue_numbers, chain.segment_numbers, strict=True):
            residue_to_segment.setdefault(int(residue_number), int(segment_number))
        residue_maps.append(residue_to_segment)
    output_lines: list[str] = []
    nc = 0
    nr = 0
    previous_residue_field = ""
    for raw_line in model.lines:
        line = raw_line.rstrip("\n").ljust(80)
        record = line[:6]
        if record == "ATOM  " and nc < len(model.chains):
            out = list(line[:80])
            chain = model.chains[nc]
            if chain.solutions is None:
                continue
            residue_field = line[17:27]
            if residue_field != previous_residue_field:
                nr += 1
                previous_residue_field = residue_field
            segment = residue_maps[nc].get(nr, 1)
            segment = min(max(segment, 1), chain.n_segments)
            solution = chain.solutions[segment - 1] if chain.solutions is not None else np.zeros(6, dtype=np.float32)
            rot = euler_to_matrix(solution[:3])
            coord = _parse_xyz(line)
            moved = rot @ (coord - chain.centroid) + chain.centroid + solution[3:6]
            out[30:54] = list(f"{moved[0]:8.3f}{moved[1]:8.3f}{moved[2]:8.3f}")
            output_lines.append("".join(out).rstrip())
        elif line.startswith("TER"):
            write_ter = nc < len(model.chains) and model.chains[nc].solutions is not None
            nc += 1
            previous_residue_field = ""
            if write_ter:
                output_lines.append("TER")
        else:
            output_lines.append(line.rstrip())
    _write_structure_lines(output_path, output_lines)


def write_top_pose_bundle_pdb(
    model: PDBModel,
    poses: list[np.ndarray],
    output_path: str | Path,
    chain_labels: str = ALPHABET,
) -> None:
    output_path = Path(output_path)
    if not model.chains:
        raise ValueError("Model does not contain any chains.")
    if len(model.chains) != 1:
        raise ValueError("write_top_pose_bundle_pdb currently expects a single-chain PDB model.")

    chain = model.chains[0]
    output_lines: list[str] = []
    for pose_idx, pose in enumerate(poses):
        chain_id = chain_labels[pose_idx % len(chain_labels)]
        solution = np.asarray(pose, dtype=np.float32)
        for raw_line in model.lines:
            line = raw_line.rstrip("\n").ljust(80)
            record = line[:6]
            if record == "ATOM  ":
                out = list(line[:80])
                rot = euler_to_matrix(solution[:3])
                coord = _parse_xyz(line)
                moved = rot @ (coord - chain.centroid) + chain.centroid + solution[3:6]
                out[21] = chain_id
                out[30:54] = list(f"{moved[0]:8.3f}{moved[1]:8.3f}{moved[2]:8.3f}")
                output_lines.append("".join(out).rstrip())
            elif line.startswith("REMARK LINK"):
                output_lines.append(line.rstrip())
            elif record == "TER   ":
                output_lines.append("TER")

    _write_structure_lines(output_path, output_lines)
