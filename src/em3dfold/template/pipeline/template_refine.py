from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from em3dfold.io.fileio import extract_lines_by_ca, getlines
from em3dfold.io.pdbio import read_pdb
from em3dfold.io.seqio import get_sequence_from_pdb_lines, nwalign_fast, read_fasta
from em3dfold.polymer_utils.residue_constants import restype_3_to_index
from em3dfold.template.utils.domain import (
    convert_domains_to_1d_repr,
    parse_unidoc_result,
    run_unidoc,
)
from em3dfold.utils.misc_utils import abspath, pjoin


AlignmentTuple = Tuple[str, str, str, float, float, float, float]


@dataclass
class StructureModel:
    path: str
    atom_pos: np.ndarray
    atom_mask: np.ndarray
    res_type: np.ndarray
    res_idx: np.ndarray
    chain_idx: np.ndarray
    raw_lines: List[str]
    ca_lines: List[str]
    seq: str
    align_seq: str


@dataclass
class ChainTemplateMatch:
    chain_index: int
    template_index: int
    alignment: Optional[AlignmentTuple]
    seqid: float
    seqcov: float


@dataclass
class TemplateDomainInfo:
    template_index: int
    domains: List[List[List[int]]]
    domains_1d: np.ndarray


@dataclass
class TemplateRefineContext:
    chains: List[StructureModel]
    templates: List[StructureModel]
    target_seqs: List[str]
    best_matches: Dict[int, ChainTemplateMatch]
    all_pair_alignments: Dict[Tuple[int, int], AlignmentTuple]
    template_domains: Dict[int, TemplateDomainInfo]
    lib_dir: str
    work_dir: str
    verbose: bool


@dataclass
class ChainStructureBundle:
    atom_pos: List[np.ndarray]
    atom_mask: List[np.ndarray]
    res_type: List[np.ndarray]
    res_idx: Optional[List[np.ndarray]] = None


@dataclass
class FixPassResult:
    templ_bundle: ChainStructureBundle
    chain_templ_bundle: ChainStructureBundle


@dataclass
class ImpPassResult:
    untrimmed_bundle: ChainStructureBundle
    trimmed_bundle: ChainStructureBundle


def _sanitize_align_seq(seq: str) -> str:
    return "".join("G" if ch == "X" else ch for ch in seq)


def is_valid_alignment_result(result) -> bool:
    if result is None or len(result) < 7:
        return False
    return all(isinstance(result[i], str) for i in [0, 1, 2])


def _is_pdb_protein_atom_line(line: str) -> bool:
    if not line.startswith("ATOM"):
        return True
    try:
        resname = line[17:20].strip()
        return int(restype_3_to_index[resname]) < 20
    except Exception:
        return False


def load_structure_models(
    paths: List[str],
    protein_only: bool = False,
    skip_empty: bool = False,
) -> List[StructureModel]:
    models = []
    for path in paths:
        norm_path = abspath(path)
        raw_lines = getlines(norm_path)
        atom_pos, atom_mask, res_type, res_idx, chain_idx = read_pdb(
            norm_path,
            keep_valid=False,
        )

        if protein_only:
            protein_mask = res_type < 20
            atom_pos = atom_pos[protein_mask]
            atom_mask = atom_mask[protein_mask]
            res_type = res_type[protein_mask]
            res_idx = res_idx[protein_mask]
            chain_idx = chain_idx[protein_mask]
            if norm_path.lower().endswith(".pdb"):
                raw_lines = [line for line in raw_lines if _is_pdb_protein_atom_line(line)]

        if skip_empty and len(atom_pos) == 0:
            continue

        ca_lines = extract_lines_by_ca(raw_lines)
        seq = get_sequence_from_pdb_lines(raw_lines)
        if skip_empty and len(seq) == 0:
            continue

        models.append(
            StructureModel(
                path=norm_path,
                atom_pos=atom_pos,
                atom_mask=atom_mask,
                res_type=res_type,
                res_idx=res_idx,
                chain_idx=chain_idx,
                raw_lines=raw_lines,
                ca_lines=ca_lines,
                seq=seq,
                align_seq=_sanitize_align_seq(seq),
            )
        )
    return models


def load_chain_models(paths: List[str]) -> List[StructureModel]:
    return load_structure_models(paths, protein_only=False, skip_empty=False)


def load_template_models(paths: List[str]) -> List[StructureModel]:
    return load_structure_models(paths, protein_only=True, skip_empty=True)


def load_target_sequences(seq_path: Optional[str]) -> List[str]:
    if seq_path is None:
        return []
    return read_fasta(abspath(seq_path))


def build_chain_template_alignment_matrix(
    chains: List[StructureModel],
    templates: List[StructureModel],
    lib_dir: str,
    temp_dir: str,
    verbose: bool,
    debug: bool,
) -> Tuple[Dict[Tuple[int, int], AlignmentTuple], Dict[int, ChainTemplateMatch]]:
    seq_temp_dir = pjoin(temp_dir, "seqs")
    os.makedirs(seq_temp_dir, exist_ok=True)

    pair_alignments: Dict[Tuple[int, int], AlignmentTuple] = {}
    best_matches: Dict[int, ChainTemplateMatch] = {}

    for i, chain in enumerate(chains):
        best_match = ChainTemplateMatch(
            chain_index=i,
            template_index=-1,
            alignment=None,
            seqid=-1e6,
            seqcov=-1e6,
        )
        if len(chain.align_seq) == 0:
            best_matches[i] = best_match
            continue

        for k, template in enumerate(templates):
            if len(template.align_seq) == 0:
                continue

            result = nwalign_fast(
                chain.align_seq,
                template.align_seq,
                lib_dir=lib_dir,
                temp_dir=seq_temp_dir,
                verbose=verbose,
                namea=f"chain_{i}",
                nameb=f"templ_{k}",
                debug=debug,
            )
            if not is_valid_alignment_result(result):
                continue

            pair_alignments[(i, k)] = result
            seqid = float(result[3])
            seqcov = float(result[4])
            if seqid > best_match.seqid:
                best_match = ChainTemplateMatch(
                    chain_index=i,
                    template_index=k,
                    alignment=result,
                    seqid=seqid,
                    seqcov=seqcov,
                )

        best_matches[i] = best_match

    return pair_alignments, best_matches


def prepare_template_domains(
    templates: List[StructureModel],
    lib_dir: str,
    temp_dir: str,
    verbose: bool,
) -> Dict[int, TemplateDomainInfo]:
    template_domains: Dict[int, TemplateDomainInfo] = {}
    for i, template in enumerate(templates):
        unidoc_result = run_unidoc(
            template.path,
            chain="A",
            lib_dir=lib_dir,
            temp_dir=temp_dir,
            verbose=verbose,
            domain_type="unmerged",
        )
        domains = parse_unidoc_result(unidoc_result)
        domains_1d = convert_domains_to_1d_repr(domains)
        template_domains[i] = TemplateDomainInfo(
            template_index=i,
            domains=domains,
            domains_1d=domains_1d,
        )
    return template_domains


def build_template_refine_context(
    chain_paths: List[str],
    template_paths: List[str],
    lib_dir: str,
    work_dir: str,
    seq_path: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    prepare_domains_flag: bool = False,
) -> TemplateRefineContext:
    lib_dir = abspath(lib_dir)
    work_dir = abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    chains = load_chain_models(chain_paths)
    templates = load_template_models(template_paths)
    target_seqs = load_target_sequences(seq_path)
    all_pair_alignments, best_matches = build_chain_template_alignment_matrix(
        chains=chains,
        templates=templates,
        lib_dir=lib_dir,
        temp_dir=work_dir,
        verbose=verbose,
        debug=debug,
    )

    template_domains: Dict[int, TemplateDomainInfo] = {}
    if prepare_domains_flag:
        template_domains = prepare_template_domains(
            templates=templates,
            lib_dir=lib_dir,
            temp_dir=work_dir,
            verbose=verbose,
        )

    return TemplateRefineContext(
        chains=chains,
        templates=templates,
        target_seqs=target_seqs,
        best_matches=best_matches,
        all_pair_alignments=all_pair_alignments,
        template_domains=template_domains,
        lib_dir=lib_dir,
        work_dir=work_dir,
        verbose=verbose,
    )


def bundle_from_models(models: List[StructureModel]) -> ChainStructureBundle:
    return ChainStructureBundle(
        atom_pos=[model.atom_pos for model in models],
        atom_mask=[model.atom_mask for model in models],
        res_type=[model.res_type for model in models],
        res_idx=[model.res_idx for model in models],
    )


def write_chain_structure_bundle(filename: str, bundle: ChainStructureBundle):
    kwargs = dict(
        filename=filename,
        chains_atom_pos=bundle.atom_pos,
        chains_atom_mask=bundle.atom_mask,
        chains_res_type=bundle.res_type,
        suffix=os.path.splitext(filename)[1].lstrip("."),
    )
    if bundle.res_idx is not None:
        kwargs["chains_res_idx"] = bundle.res_idx
    from em3dfold.io.pdbio import chains_atom_pos_to_pdb

    chains_atom_pos_to_pdb(**kwargs)


def write_imp_outputs(
    out_dir: str,
    untrimmed_bundle: ChainStructureBundle,
    trimmed_bundle: ChainStructureBundle,
    fallback: bool = False,
):
    untrimmed_path = pjoin(out_dir, "imp_chains_untrimmed.cif")
    write_chain_structure_bundle(untrimmed_path, untrimmed_bundle)
    if fallback:
        print(f"# Write original chains to {untrimmed_path}")
    else:
        print(f"# Write untrimmed chains to {untrimmed_path}")

    trimmed_path = pjoin(out_dir, "imp_chains_trimmed.cif")
    write_chain_structure_bundle(trimmed_path, trimmed_bundle)
    if fallback:
        print(f"# Write original chains to {trimmed_path}")
    else:
        print(f"# Write trimmed chains to {trimmed_path}")


def write_fix_outputs(
    out_dir: str,
    templ_bundle: Optional[ChainStructureBundle] = None,
    chain_templ_bundle: Optional[ChainStructureBundle] = None,
    fallback: bool = False,
):
    if templ_bundle is not None:
        templ_path = pjoin(out_dir, "fix_templs.cif")
        write_chain_structure_bundle(templ_path, templ_bundle)
        if fallback:
            print(f"# Write original chains to {templ_path}")
        else:
            print(f"# Write fixed templates to {templ_path}")

    if chain_templ_bundle is not None:
        chain_templ_path = pjoin(out_dir, "fix_chains_templs.cif")
        write_chain_structure_bundle(chain_templ_path, chain_templ_bundle)
        if fallback:
            print(f"# Write original chains to {chain_templ_path}")
        else:
            print(f"# Write fixed templates to {chain_templ_path}")
