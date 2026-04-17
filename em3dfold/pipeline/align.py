import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from em3dfold.polymer_utils.polymer import POLYMER_KEYS, get_polymer_from_file_path
from em3dfold.io.pdbio import chains_atom_pos_to_pdb

DEFAULT_DISTANCE_THRESHOLD = 3.0
DEFAULT_MIN_TEMPLATE_LENGTH = 30
MAX_ASSEMBLY_STEPS = 1000


@dataclass(frozen=True)
class AlignmentResult:
    template_name: str
    tm_score: float
    coverage_score: float
    aligned_structure_path: str


def is_structure_file(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in {".cif", ".pdb"}


def sanitize_file_stem(value: str) -> str:
    sanitized = [char if char.isalnum() else "_" for char in value]
    token = "".join(sanitized).strip("_")
    return token or "structure"


def slice_protein(protein, slice_array: np.ndarray):
    num_residues = len(protein.aatype)
    for key in POLYMER_KEYS:
        value = getattr(protein, key)
        if hasattr(value, "shape") and value.shape[0] == num_residues:
            setattr(protein, key, value[slice_array])
    return protein


def write_single_chain_protein(protein, output_path: str) -> str:
    chains_atom_pos_to_pdb(
        output_path,
        chains_atom_pos=[protein.atom_positions],
        chains_atom_mask=[protein.atom_mask],
        chains_res_type=[protein.aatype],
        chains_bfactor=[protein.b_factors],
        chains_res_idx=None,
    )
    return output_path


def create_working_model(
    model_path: str,
    output_dir: str,
) -> str:
    protein = get_polymer_from_file_path(model_path)
    model_name = os.path.basename(os.path.abspath(output_dir))
    working_model_path = os.path.join(output_dir, f"{model_name}_raw.cif")
    return write_single_chain_protein(protein, working_model_path)


def list_template_files(
    template_dir: str,
    min_template_length: int = DEFAULT_MIN_TEMPLATE_LENGTH,
) -> List[str]:
    template_paths = []
    for template_name in sorted(os.listdir(template_dir)):
        if not is_structure_file(template_name):
            continue
        template_path = os.path.join(template_dir, template_name)
        protein = get_polymer_from_file_path(template_path)
        if len(protein.aatype) <= min_template_length:
            continue
        template_paths.append(template_path)
    return template_paths


def parse_tm_score(usalign_stdout: str) -> float:
    tm_scores = []
    for line in usalign_stdout.splitlines():
        if line.startswith("TM-score= "):
            tm_scores.append(float(line[10:17]))
    return tm_scores[0] if tm_scores else 0.0


def resolve_aligned_output_path(output_prefix: str, template_path: str) -> str:
    template_suffix = Path(template_path).suffix.lower()
    preferred_path = output_prefix + template_suffix
    if os.path.exists(preferred_path):
        return preferred_path

    fallback_path = os.path.join(
        os.path.dirname(output_prefix),
        os.path.basename(template_path),
    )
    if os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(
        f"Unable to find the aligned output for template: {template_path}"
    )


def compute_coverage_score(
    aligned_template_path: str,
    target_model_path: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> float:
    aligned_template = get_polymer_from_file_path(aligned_template_path)
    target_model = get_polymer_from_file_path(target_model_path)

    aligned_ca_positions = aligned_template.atom_positions[:, 1]
    target_ca_positions = target_model.atom_positions[:, 1]
    if len(aligned_ca_positions) == 0 or len(target_ca_positions) == 0:
        return 0.0

    target_tree = cKDTree(target_ca_positions)
    distances, _ = target_tree.query(aligned_ca_positions, k=1)
    return float(np.sum(distances < distance_threshold) / np.sqrt(len(distances)))


def align_template(
    template_path: str,
    target_model_path: str,
    output_dir: str,
    align_script: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> AlignmentResult:
    template_name = os.path.basename(template_path)
    output_prefix = os.path.join(output_dir, sanitize_file_stem(Path(template_name).stem))
    command = [
        align_script,
        template_path,
        target_model_path,
        "-mm",
        "1",
        "-ter",
        "0",
        "-o",
        output_prefix,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"USalign failed for {template_name}:\n{completed.stderr or completed.stdout}"
        )

    aligned_structure_path = resolve_aligned_output_path(output_prefix, template_path)
    return AlignmentResult(
        template_name=template_name,
        tm_score=parse_tm_score(completed.stdout),
        coverage_score=compute_coverage_score(
            aligned_template_path=aligned_structure_path,
            target_model_path=target_model_path,
            distance_threshold=distance_threshold,
        ),
        aligned_structure_path=aligned_structure_path,
    )


def remove_overlapping_residues(
    reference_structure_path: Optional[str],
    target_model_path: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> str:
    target_protein = get_polymer_from_file_path(target_model_path)
    if reference_structure_path is None:
        return write_single_chain_protein(target_protein, target_model_path)

    reference_protein = get_polymer_from_file_path(reference_structure_path)
    reference_ca_positions = reference_protein.atom_positions[:, 1]
    target_ca_positions = target_protein.atom_positions[:, 1]
    if len(reference_ca_positions) == 0 or len(target_ca_positions) == 0:
        return write_single_chain_protein(target_protein, target_model_path)

    reference_tree = cKDTree(reference_ca_positions)
    distances, _ = reference_tree.query(target_ca_positions, k=1)
    remaining_mask = distances > distance_threshold
    remaining_protein = slice_protein(target_protein, remaining_mask)
    return write_single_chain_protein(remaining_protein, target_model_path)


def choose_best_alignment(
    alignment_results: List[AlignmentResult],
    tm_score_threshold: float,
) -> Optional[AlignmentResult]:
    if not alignment_results:
        return None

    if tm_score_threshold < 1:
        eligible_results = [
            result for result in alignment_results if result.tm_score > tm_score_threshold
        ]
        if not eligible_results:
            return None
        return max(eligible_results, key=lambda result: result.coverage_score)

    best_result = max(alignment_results, key=lambda result: result.coverage_score)
    if best_result.tm_score < 0:
        return None
    return best_result


def merge_selected_templates(aligned_paths: List[str], output_path: str) -> str:
    if not aligned_paths:
        raise ValueError("No aligned templates were provided for merging.")

    aatype_list = []
    atom_positions_list = []
    atom_mask_list = []
    b_factors_list = []
    for aligned_path in aligned_paths:
        protein = get_polymer_from_file_path(aligned_path)
        aatype_list.append(protein.aatype)
        atom_positions_list.append(protein.atom_positions)
        atom_mask_list.append(protein.atom_mask)
        b_factors_list.append(protein.b_factors)

    chains_atom_pos_to_pdb(
        output_path,
        chains_atom_pos=atom_positions_list,
        chains_atom_mask=atom_mask_list,
        chains_res_type=aatype_list,
        chains_bfactor=b_factors_list,
        chains_res_idx=None,
    )

    return output_path


def assemble_with_templates(
    model_path: str,
    template_dir: str,
    output_dir: str,
    align_script: str,
    tm_score_threshold: float = 1,
    min_template_length: int = DEFAULT_MIN_TEMPLATE_LENGTH,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> str:
    output_dir = os.path.abspath(output_dir)
    template_dir = os.path.abspath(template_dir)

    steps_dir = os.path.join(output_dir, "all_steps")
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(steps_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    remaining_model_path = create_working_model(
        model_path=model_path,
        output_dir=output_dir,
    )
    template_paths = list_template_files(
        template_dir=template_dir,
        min_template_length=min_template_length,
    )
    template_use_count = defaultdict(int)
    selected_alignment_paths = []

    for step_index in range(MAX_ASSEMBLY_STEPS):
        alignment_results = []
        for template_path in template_paths:
            template_name = os.path.basename(template_path)
            if tm_score_threshold >= 1 and template_use_count[template_name] > 0:
                continue
            alignment_results.append(
                align_template(
                    template_path=template_path,
                    target_model_path=remaining_model_path,
                    output_dir=temp_dir,
                    align_script=align_script,
                    distance_threshold=distance_threshold,
                )
            )

        best_alignment = choose_best_alignment(
            alignment_results=alignment_results,
            tm_score_threshold=tm_score_threshold,
        )
        if best_alignment is None:
            break

        template_use_count[best_alignment.template_name] += 1
        step_file_name = f"step_{step_index:04d}_{best_alignment.template_name}"
        saved_step_path = os.path.join(steps_dir, step_file_name)
        shutil.copy(best_alignment.aligned_structure_path, saved_step_path)
        selected_alignment_paths.append(saved_step_path)
        remove_overlapping_residues(
            reference_structure_path=saved_step_path,
            target_model_path=remaining_model_path,
            distance_threshold=distance_threshold,
        )

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(model_path)),
        "model_assembly.cif",
    )
    if not selected_alignment_paths:
        shutil.copy(remaining_model_path, output_path)
        return output_path

    return merge_selected_templates(selected_alignment_paths, output_path)


def add_args(parser):
    parser.add_argument(
        "--usalign",
        default="USalign",
        help="Directory to USalign program"
    )
    parser.add_argument(
        "--template-dir",
        "--td",
        "-td",
        required=True,
        help="Directory containing template structures.",
    )
    parser.add_argument(
        "--model",
        "--c",
        "-c",
        required=True,
        help="Model used as the assembly target.",
    )
    parser.add_argument(
        "--tmscore-threshold",
        "--tt",
        "-tt",
        type=float,
        default=1,
        help=(
            "Minimum TM-score for selecting a template. If the threshold is at least "
            "1, each template is used at most once."
        ),
    )
    parser.add_argument(
        "--min-template-length",
        type=int,
        default=DEFAULT_MIN_TEMPLATE_LENGTH,
        help="Ignore templates with this many residues or fewer.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help="C-alpha distance threshold used for coverage scoring and residue removal.",
    )
    return parser


def main(args):
    align_script = args.usalign
    work_dir = os.path.join(os.path.dirname(Path(args.model)), "align_result")
    work_dir_already_exists = os.path.isdir(work_dir)

    print(f"Working directory: {work_dir}")
    print("Running template assembly...")
    output_cif = assemble_with_templates(
        model_path=args.model,
        template_dir=args.template_dir,
        output_dir=work_dir,
        align_script=align_script,
        tm_score_threshold=args.tmscore_threshold,
        min_template_length=args.min_template_length,
        distance_threshold=args.distance_threshold,
    )
    if output_cif and not work_dir_already_exists and os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    print("Done.")
    print(f"The final structure is saved to: {output_cif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Assemble de novo built models with whole-template alignment.",
    )
    parsed_args = add_args(parser).parse_args()
    main(parsed_args)


