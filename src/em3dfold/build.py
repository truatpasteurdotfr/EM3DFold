"""Main program"""
import os
import sys
import time
import shutil
import argparse
import tempfile
from pathlib import Path

from em3dfold.io.pdbio import (
    chains_atom_pos_to_pdb,
    convert_to_chains,
    fix_quotes,
    read_pdb,
)
from em3dfold.io.seqio import read_fasta
from em3dfold.utils.log_utils import progress, progress_stage
from em3dfold.utils.misc_utils import pjoin, abspath
from em3dfold.utils.torch_utils import clear_cuda_cache

EM_WEIGHTS_ENV_VAR = "EM_WEIGHTS_DIR"


def add_args(parser):
    parser.add_argument("--map", "-m", help="Input map", required=True)
    # Using --dna/--rna instead of a consensus --seq
    parser.add_argument("--protein", "-p", help="Input protein sequence")
    parser.add_argument("--rna", "-r", help="Input rna sequence")
    parser.add_argument("--dna", "-d", help="Input dna sequence")
    # Add protein chain templates, e.g. predicted AlphaFold models
    parser.add_argument(
        "--protein-template",
        "-pt",
        nargs="+",
        default=None,
        help="Input protein template file(s); each protein chain will be split into temp_dir/templates/template_x_chain_x.cif",
    )
    parser.add_argument("--output", "-o", help="Output directory", required=True)
    parser.add_argument("--device", "--gpu", help="GPU device, default = '0'", default="0")
    parser.add_argument(
        "--lm-weights-dir",
        help="Optional shared directory for ESM and RiNALMo weights",
    )
    parser.add_argument(
        "--pred-weights-dir",
        "--weights-dir",
        dest="pred_weights_dir",
        help="Optional shared root directory for pred.py and inferlm_v3x weights",
    )
    parser.add_argument(
        "--temp-root",
        help="Optional parent directory for the build temporary workspace",
    )
    parser.add_argument(
        "--use-system-temp",
        action="store_true",
        help="Create the build temporary workspace with tempfile.mkdtemp instead of <output>/temp",
    )
    parser.add_argument("--keep-temp-files", action="store_true", help="Whether to keep temp files")
    # Using GPU for faster getp
    parser.add_argument("--gpu-getp", action="store_true", help="Using GPU to acclerate mean-shift for large maps")
    # Skipping controls
    skip_group = parser.add_argument_group("Skipping options")
    skip_group.add_argument("--skip-preprocess", action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-cx",         action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-denovo",     action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-map-to-p",   action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-protein", action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-na",      action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-infer-na-aa",   action='store_true', help=argparse.SUPPRESS)
    # With templates
    skip_group.add_argument("--skip-imp",  action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-fix",  action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-fit",  action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip-assemble", action='store_true', help=argparse.SUPPRESS)
    return parser


def _run_getp_pipeline(
    map_path,
    output_dir,
    device,
    raw_output_path,
    *,
    atom_name="CA",
    res_name="GLY",
    chain_id="A",
    element=None,
    rmax,
    dmerge,
    thresh,
    ratio=0.05,
    run_getp=True,
    run_g2p=False,
):
    from em3dfold.pipeline import getp

    os.makedirs(output_dir, exist_ok=True)
    getp_args = argparse.Namespace()
    getp_args.map = map_path
    getp_args.device = device
    getp_args.p = None
    getp_args.output = output_dir
    getp_args.getp = None
    getp_args.run_g2p = run_g2p
    getp_args.run_getp = run_getp
    getp_args.ratio = ratio
    getp_args.thresh = thresh
    getp_args.res = 6.0
    getp_args.nt = 4
    getp_args.filter = 0.0
    getp_args.dmerge = dmerge
    getp_args.rmax = rmax
    getp_args.fuse_g2p = True
    getp_args.g2p_cover_distance = 1.75
    getp_args.g2p_supplement_merge_distance = 1.5
    getp_args.g2p_refine_radius = 1.5
    getp_args.g2p_refine_iters = 2
    getp_args.g2p_max_supplements = None
    getp_args.atom_name = atom_name
    getp_args.res_name = res_name
    getp_args.chain_id = chain_id
    getp_args.element = element
    getp.main(getp_args)

    merged_output_path = pjoin(output_dir, "merged.pdb")
    if os.path.exists(merged_output_path):
        shutil.copy(merged_output_path, raw_output_path)
    elif os.path.exists(pjoin(output_dir, "raw.pdb")):
        shutil.copy(pjoin(output_dir, "raw.pdb"), raw_output_path)
    else:
        raise FileNotFoundError(
            "getp did not produce merged.pdb or raw.pdb under {}".format(output_dir)
        )


def _first_existing_path(*paths):
    for path in paths:
        if path is not None and os.path.exists(path):
            return path
    return None


def _merge_initial_polymer_files(output_path, file_specs):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        atom_serial = 1
        for file_path, chain_id in file_specs:
            if file_path is None or (not os.path.exists(file_path)):
                continue
            with open(file_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.startswith(("ATOM", "HETATM")):
                        line = f"{line[:6]}{atom_serial:>5d}{line[11:]}"
                        line = f"{line[:21]}{chain_id}{line[22:]}"
                        fout.write(line)
                        atom_serial += 1
                    elif line.startswith("TER"):
                        fout.write("TER\n")
        fout.write("END\n")


def _normalize_torch_device(device):
    device = str(device)
    if device.isdigit():
        return f"cuda:{device}"
    return device


def _resolve_env_weights_root():
    env_value = os.environ.get(EM_WEIGHTS_ENV_VAR)
    if env_value is None or str(env_value).strip() == "":
        return None
    return Path(env_value).expanduser().resolve()


def _first_existing_path_obj(*paths):
    for path in paths:
        if path is not None and path.exists():
            return path.resolve()
    return None


def _resolve_pred_weights_dir(pred_weights_dir, script_dir):
    if pred_weights_dir:
        return abspath(pred_weights_dir)

    env_root = _resolve_env_weights_root()
    if env_root is not None:
        resolved = _first_existing_path_obj(
            env_root / "weights",
            env_root,
            env_root.parent / "weights" if env_root.name == "lm_weights" else None,
        )
        if resolved is not None:
            return str(resolved)
        if env_root.name == "weights":
            return str(env_root)
        return str((env_root / "weights").resolve())

    return pjoin(script_dir, "weights")


def _resolve_lm_weights_dir(lm_weights_dir):
    if lm_weights_dir:
        return Path(lm_weights_dir).expanduser().resolve()

    env_root = _resolve_env_weights_root()
    if env_root is None:
        return None

    resolved = _first_existing_path_obj(
        env_root / "lm_weights",
        env_root if env_root.name == "lm_weights" else None,
        env_root.parent / "lm_weights" if env_root.name == "weights" else None,
    )
    if resolved is not None:
        return resolved

    if env_root.name == "weights":
        return (env_root.parent / "lm_weights").resolve()
    return (env_root / "lm_weights").resolve()


def _prepare_temp_dir(out_dir, temp_root=None, use_system_temp=False):
    if use_system_temp:
        parent_dir = abspath(temp_root) if temp_root else None
        temp_dir = tempfile.mkdtemp(prefix="em3dfold_", dir=parent_dir)
    else:
        if temp_root:
            temp_dir = pjoin(abspath(temp_root), os.path.basename(out_dir), "temp")
        else:
            temp_dir = pjoin(out_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
    return abspath(temp_dir)


def _extract_protein_template_chains(template_paths, temp_dir):
    import numpy as np

    template_dir = pjoin(temp_dir, "templates")
    if os.path.exists(template_dir):
        shutil.rmtree(template_dir)
    os.makedirs(template_dir, exist_ok=True)

    written_paths = []
    for template_idx, template_path in enumerate(template_paths):
        template_path = abspath(template_path)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Protein template file not found: {template_path}")

        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
            template_path,
            keep_valid=False,
            return_bfactor=True,
        )

        protein_mask = res_type < 20
        if not np.any(protein_mask):
            print(f"# No protein residues found in template {template_path}, skip")
            continue

        atom_pos = atom_pos[protein_mask]
        atom_mask = atom_mask[protein_mask]
        res_type = res_type[protein_mask]
        res_idx = res_idx[protein_mask]
        chain_idx = chain_idx[protein_mask]
        bfactor = bfactor[protein_mask]

        unique_chain_indices = np.unique(chain_idx)
        for chain_local_idx, source_chain_idx in enumerate(unique_chain_indices):
            chain_mask = chain_idx == source_chain_idx
            output_cif_path = pjoin(
                template_dir,
                f"template_{template_idx}_chain_{chain_local_idx}.cif",
            )
            output_pdb_path = pjoin(
                template_dir,
                f"template_{template_idx}_chain_{chain_local_idx}.pdb",
            )
            chains_atom_pos_to_pdb(
                output_cif_path,
                chains_atom_pos=[atom_pos[chain_mask]],
                chains_atom_mask=[atom_mask[chain_mask]],
                chains_res_type=[res_type[chain_mask]],
                chains_res_idx=[res_idx[chain_mask]],
                chains_idx=[0],
                chains_bfactor=[bfactor[chain_mask]],
                suffix="cif",
            )
            chains_atom_pos_to_pdb(
                output_pdb_path,
                chains_atom_pos=[atom_pos[chain_mask]],
                chains_atom_mask=[atom_mask[chain_mask]],
                chains_res_type=[res_type[chain_mask]],
                chains_res_idx=[res_idx[chain_mask]],
                chains_idx=[0],
                chains_bfactor=[bfactor[chain_mask]],
                suffix="pdb",
            )
            written_paths.append(output_pdb_path)
            print(f"# Write protein template chain to {output_cif_path}")
            print(f"# Write protein template chain to {output_pdb_path}")

    if written_paths:
        print(f"# Extracted {len(written_paths)} protein template chains into {template_dir}")
    else:
        print("# No protein template chains were extracted")
    return written_paths


def _split_structure_to_chains(structure_path, output_dir, *, protein_only=False, suffix="pdb"):
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
        structure_path,
        keep_valid=False,
        return_bfactor=True,
    )

    if protein_only:
        protein_mask = res_type < 20
        if not np.any(protein_mask):
            return []
        atom_pos = atom_pos[protein_mask]
        atom_mask = atom_mask[protein_mask]
        res_type = res_type[protein_mask]
        res_idx = res_idx[protein_mask]
        chain_idx = chain_idx[protein_mask]
        bfactor = bfactor[protein_mask]

    (
        chains_atom_pos,
        chains_atom_mask,
        chains_res_type,
        chains_res_idx,
        chains_bfactor,
    ) = convert_to_chains(
        chain_idx,
        atom_pos,
        atom_mask,
        res_type,
        res_idx,
        bfactor,
    )

    output_paths = []
    for chain_local_idx in range(len(chains_atom_pos)):
        output_path = pjoin(output_dir, f"chain_{chain_local_idx}.{suffix}")
        chains_atom_pos_to_pdb(
            output_path,
            chains_atom_pos=[chains_atom_pos[chain_local_idx]],
            chains_atom_mask=[chains_atom_mask[chain_local_idx]],
            chains_res_type=[chains_res_type[chain_local_idx]],
            chains_res_idx=[chains_res_idx[chain_local_idx]],
            chains_idx=[0],
            chains_bfactor=[chains_bfactor[chain_local_idx]],
            suffix=suffix,
        )
        output_paths.append(output_path)
    return output_paths


def _has_cli_sequence_arg(input_seq_path):
    return input_seq_path is not None and str(input_seq_path).strip() != ""


def _resolve_nonempty_seq_path(cli_has_seq_arg, formatted_seq_path, label):
    if not cli_has_seq_arg:
        return None
    if not os.path.exists(formatted_seq_path):
        print(f"# {label} sequence file not found at {formatted_seq_path}, skip {label}")
        return None

    try:
        seqs = [seq.strip() for seq in read_fasta(formatted_seq_path) if seq.strip()]
    except Exception as exc:
        print(f"# Failed to read {label} sequence file {formatted_seq_path}: {exc}")
        return None

    if not seqs:
        print(f"# {label} sequence file is empty at {formatted_seq_path}, skip {label}")
        return None

    if sum(len(seq) for seq in seqs) <= 0:
        print(f"# {label} sequence length is zero at {formatted_seq_path}, skip {label}")
        return None

    return formatted_seq_path


def _load_esm_model(device, lm_weights_dir=None):
    import torch
    import esm

    shared_dir = _resolve_lm_weights_dir(lm_weights_dir)
    if shared_dir is not None:
        local_ckpt = shared_dir / "esm2_t33_650M_UR50D.pt"
        if local_ckpt.exists() and hasattr(esm.pretrained, "load_model_and_alphabet_local"):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(local_ckpt))
        else:
            # Fall back to torch hub cache layout under the shared directory.
            os.environ["TORCH_HOME"] = str(shared_dir)
            torch.hub.set_dir(str(shared_dir))
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    model = model.eval().to(device)
    return model, alphabet


def _load_rinalmo_model(device, lm_weights_dir=None):
    import importlib
    import torch

    local_rinalmo_dir = abspath(
        pjoin(os.path.dirname(__file__), "rinalmo", "RiNALMo-1.0")
    )
    if not os.path.isdir(local_rinalmo_dir):
        raise FileNotFoundError(
            f"Local RiNALMo directory not found: {local_rinalmo_dir}"
        )
    if local_rinalmo_dir not in sys.path:
        sys.path.insert(0, local_rinalmo_dir)

    pretrained = importlib.import_module("rinalmo.pretrained")
    shared_dir = _resolve_lm_weights_dir(lm_weights_dir)
    pretrained_weights_path = None
    if shared_dir is not None:
        candidate_paths = [
            shared_dir / "rinalmo_giga_pretrained.pt",
            shared_dir / "giga-v1.pt",
            shared_dir / "rinalmo" / "rinalmo_giga_pretrained.pt",
            shared_dir / "rinalmo" / "giga-v1.pt",
        ]
        pretrained_weights_path = next(
            (path for path in candidate_paths if path.exists()),
            None,
        )
        if pretrained_weights_path is None:
            pretrained.DEFAULT_CACHE_DIR = shared_dir / "rinalmo"
        else:
            config = pretrained.model_config("giga")
            model = pretrained.RiNALMo(config)
            alphabet = pretrained.Alphabet(**config["alphabet"])
            model.load_state_dict(torch.load(pretrained_weights_path, map_location="cpu"))
            model = model.to(device)
            model.eval()
            return model, alphabet

    model, alphabet = pretrained.get_pretrained_model(model_name="giga-v1")
    model = model.to(device)
    model.eval()
    return model, alphabet


def _build_protein_lm(seq_path, output_path, device, max_chain_length=1000, lm_weights_dir=None):
    from em3dfold.pipeline import get_lm
    import numpy as np

    device = _normalize_torch_device(device)
    sequences = get_lm.read_fasta(seq_path)
    filtered_sequences = []
    for sequence in sequences:
        filtered = [ch for ch in sequence if ch in get_lm.prot_restype1 and ch != "1"]
        filtered = "".join(filtered)
        if len(filtered) > 2:
            filtered_sequences.append(filtered)

    if not filtered_sequences:
        lm_embeddings = np.zeros((3, 1280), dtype=np.float32)
    else:
        model, alphabet = _load_esm_model(device, lm_weights_dir=lm_weights_dir)
        batch_converter = alphabet.get_batch_converter()
        lm_embeddings = get_lm.get_lm_embeddings(
            model,
            batch_converter,
            filtered_sequences,
            max_chain_length=max_chain_length,
        )
    np.save(output_path, lm_embeddings)
    return output_path


def _build_na_lm(
    rna_seq_path,
    dna_seq_path,
    output_path,
    device,
    max_chain_length=1000,
    lm_weights_dir=None,
):
    import numpy as np

    from em3dfold.pipeline import get_lm_na
    device = _normalize_torch_device(device)
    model, alphabet = _load_rinalmo_model(device, lm_weights_dir=lm_weights_dir)

    all_embeddings = []
    if rna_seq_path is not None and os.path.exists(rna_seq_path):
        rna_sequences = get_lm_na.filter_seqs(get_lm_na.read_fasta(rna_seq_path))
        if rna_sequences:
            all_embeddings.append(
                get_lm_na.get_lm_embeddings(
                    lang_model=model,
                    alphabet=alphabet,
                    sequences=rna_sequences,
                    max_chain_length=max_chain_length,
                )
            )

    if dna_seq_path is not None and os.path.exists(dna_seq_path):
        dna_sequences = get_lm_na.filter_seqs(get_lm_na.read_fasta(dna_seq_path))
        dna_sequences = [seq.replace("T", "U") for seq in dna_sequences]
        if dna_sequences:
            all_embeddings.append(
                get_lm_na.get_lm_embeddings(
                    lang_model=model,
                    alphabet=alphabet,
                    sequences=dna_sequences,
                    max_chain_length=max_chain_length,
                )
            )

    if all_embeddings:
        lm_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        lm_embeddings = np.zeros((3, 1280), dtype=np.float32)
    np.save(output_path, lm_embeddings)
    return output_path

def main(args):
    progress_stage("preprocess")

    script_dir = os.path.dirname(__file__)
    inferlm_v3x_model_config = pjoin(script_dir, "infer", "config", "model_v3x.yaml")
    weights_root_dir = _resolve_pred_weights_dir(args.pred_weights_dir, script_dir)
    pred_weights_dir = weights_root_dir
    all_atom_weights_dir = pjoin(weights_root_dir, "cpx", "model_all_atom")

    out_dir = abspath(args.output)
    temp_dir = _prepare_temp_dir(
        out_dir,
        temp_root=args.temp_root,
        use_system_temp=args.use_system_temp,
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"# Pred weights dir: {pred_weights_dir}")
    print(f"# inferlm weights dir: {all_atom_weights_dir}")
    print(f"# Temp root dir: {temp_dir}")

    template_chain_paths = []
    if args.protein_template:
        template_chain_paths = _extract_protein_template_chains(args.protein_template, temp_dir)

    has_protein_arg = _has_cli_sequence_arg(args.protein)
    has_rna_arg = _has_cli_sequence_arg(args.rna)
    has_dna_arg = _has_cli_sequence_arg(args.dna)
    if not (has_protein_arg or has_rna_arg or has_dna_arg):
        raise ValueError(
            "Please provide at least one input sequence via --protein and/or --rna and/or --dna."
        )

    # preprocess
    if not args.skip_preprocess:
        start = time.time()
        from em3dfold.pipeline import preprocess
        preprocess_args = argparse.Namespace()
        preprocess_args.map = args.map
        preprocess_args.protein = args.protein
        preprocess_args.rna = args.rna
        preprocess_args.dna = args.dna
        preprocess_args.output = temp_dir

        preprocess_args.device = args.device

        preprocess.main(preprocess_args)
        end = time.time()
        print("# Time = {:.4f}".format(end - start))
    else:
        progress("Skip preprocess")


    protein_seq_path = _resolve_nonempty_seq_path(
        has_protein_arg,
        pjoin(temp_dir, "format_seq_protein.fasta"),
        "Protein",
    )
    rna_seq_path = _resolve_nonempty_seq_path(
        has_rna_arg,
        pjoin(temp_dir, "format_seq_rna.fasta"),
        "RNA",
    )
    dna_seq_path = _resolve_nonempty_seq_path(
        has_dna_arg,
        pjoin(temp_dir, "format_seq_dna.fasta"),
        "DNA",
    )
    run_protein_input = protein_seq_path is not None
    run_nucleic_input = (rna_seq_path is not None) or (dna_seq_path is not None)

    # run segmentation
    if not args.skip_cx:
        progress_stage("pred")
        run_nucleic = run_nucleic_input
        run_protein = run_protein_input

        start = time.time()
        from em3dfold.pipeline import pred

        pred_args = argparse.Namespace()
        pred_args.input = pjoin(temp_dir, "format_map.mrc")
        pred_args.output = pjoin(temp_dir, "pred")
        pred_args.contour = 1e-6
        pred_args.batchsize = 40
        pred_args.device = args.device
        pred_args.model = pred_weights_dir
        pred_args.stride = 16 # 12
        pred_args.protein = run_protein
        pred_args.nucleic = run_nucleic

        pred.main(pred_args)
        clear_cuda_cache(args.device, note="pred")
        end = time.time()

        print("# Time = {:.4f}".format(end - start))
    else:
        progress("Skip dl cx")


    # ms
    if not args.skip_denovo:
        progress_stage("denovo")
        if not args.skip_map_to_p:
            ############
            # Handle C4'
            ############
            if run_nucleic_input:
                start = time.time()
                _run_getp_pipeline(
                    map_path=pjoin(temp_dir, "pred", "c4.mrc"),
                    output_dir=pjoin(temp_dir, "pred", "c4_getp"),
                    device=args.device,
                    raw_output_path=pjoin(temp_dir, "pred", "raw_c4.pdb"),
                    atom_name="C4'",
                    res_name=("A" if rna_seq_path is not None else "DA"),
                    chain_id="B",
                    element="C",
                    rmax=10.0,
                    dmerge=3.0,
                    thresh=8.0,
                    run_getp=True,
                    run_g2p=False,
                )
                end = time.time()
                print("# Time = {:.4f}".format(end - start))


            ###############
            # handle Calpha
            ###############
            if run_protein_input:
                start = time.time()
                _run_getp_pipeline(
                    map_path=pjoin(temp_dir, "pred", "ca.mrc"),
                    output_dir=pjoin(temp_dir, "pred", "ca_getp"),
                    device=args.device,
                    raw_output_path=pjoin(temp_dir, "pred", "raw_ca.pdb"),
                    atom_name="CA",
                    res_name="GLY",
                    chain_id="A",
                    element="C",
                    rmax=1.0,
                    dmerge=1.0,
                    thresh=15.0,
                    ratio=0.05,
                    run_getp=True,
                    run_g2p=True,
                )
                end = time.time()
                print("# Time = {:.4f}".format(end - start))
        else:
            progress("Skip map to p")


        run_protein = run_protein_input and (not args.skip_infer_protein)
        run_na = run_nucleic_input and (not args.skip_infer_na)

        if args.skip_infer_na_aa:
            print("# skip-infer-na-aa is ignored in unified inferlm_v3x mode")

        if run_protein or run_na:
            denovo_dir = pjoin(temp_dir, "denovo")
            os.makedirs(denovo_dir, exist_ok=True)

            prot_seq_embed_path = None
            na_seq_embed_path = None

            if run_protein:
                start = time.time()
                prot_seq_embed_path = pjoin(denovo_dir, "prot_lm.npy")
                _build_protein_lm(
                    protein_seq_path,
                    prot_seq_embed_path,
                    device=args.device,
                    max_chain_length=1000,
                    lm_weights_dir=args.lm_weights_dir,
                )
                clear_cuda_cache(args.device, note="protein LM")
                end = time.time()
                print("# Time = {:.4f}".format(end - start))
            else:
                progress("Skip protein LM embedding")

            if run_na:
                start = time.time()
                na_seq_embed_path = pjoin(denovo_dir, "na_lm.npy")
                _build_na_lm(
                    rna_seq_path,
                    dna_seq_path,
                    na_seq_embed_path,
                    device=args.device,
                    max_chain_length=1000,
                    lm_weights_dir=args.lm_weights_dir,
                )
                clear_cuda_cache(args.device, note="NA LM")
                end = time.time()
                print("# Time = {:.4f}".format(end - start))
            else:
                progress("Skip nucleic-acid LM embedding")

            initial_polymer_path = pjoin(temp_dir, "pred", "raw_polymer.pdb")
            _merge_initial_polymer_files(
                initial_polymer_path,
                [
                    (pjoin(temp_dir, "pred", "raw_ca.pdb") if run_protein else None, "A"),
                    (pjoin(temp_dir, "pred", "raw_c4.pdb") if run_na else None, "B"),
                ],
            )

            start = time.time()
            from em3dfold.infer import inferlm_v3x
            inferlm_args = argparse.Namespace()
            inferlm_args.map = pjoin(temp_dir, "format_map.mrc")
            inferlm_args.polymer = initial_polymer_path
            inferlm_args.model_dir = all_atom_weights_dir
            inferlm_args.device = args.device
            inferlm_args.crop_length = 200 if run_protein else 200
            inferlm_args.repeat_per_residue = 1
            inferlm_args.run_iters = 3
            inferlm_args.batch_size = 1
            inferlm_args.fp16 = False
            inferlm_args.voxel_size = 1.0
            inferlm_args.refine = False
            inferlm_args.no_use_random_affine = False
            inferlm_args.recycle = 4 if run_protein else 3
            inferlm_args.prot_seq_embed = prot_seq_embed_path
            inferlm_args.na_seq_embed = na_seq_embed_path
            inferlm_args.na_aa_logits = pjoin(temp_dir, "pred", "logits.npz") if run_na else None
            inferlm_args.output_dir = denovo_dir
            inferlm_args.protein_seq = protein_seq_path
            inferlm_args.dna_seq = dna_seq_path
            inferlm_args.rna_seq = rna_seq_path
            inferlm_args.min_na_chain_len = 3
            inferlm_args.fallback_to_predicted_na_types = True
            inferlm_args.pass_prev_aa_probs = True
            inferlm_args.pass_prev_rmsd = True
            inferlm_args.pass_prev_node = True
            inferlm_args.model_config = inferlm_v3x_model_config
            inferlm_v3x.main(inferlm_args)
            clear_cuda_cache(args.device, note="inferlm_v3x")
            end = time.time()
            print("# Time = {:.4f}".format(end - start))
        else:
            progress("Skip unified inferlm_v3x")

    else:
        progress("Skip denovo modeling")

    final_denovo = _first_existing_path(
        pjoin(temp_dir, "denovo", "output.cif"),
        pjoin(temp_dir, "denovo", "denovo.cif"),
    )
    final_entropy = _first_existing_path(
        pjoin(temp_dir, "denovo", "output_entropy_score.cif"),
    )
    fo = pjoin(out_dir, "output.cif")
    fo_entropy = pjoin(out_dir, "output_entropy_score.cif")

    if final_entropy is not None and os.path.exists(final_entropy):
        shutil.copy(final_entropy, fo_entropy)
        fix_quotes(fo_entropy)

    fix_output_dir = None
    imp_output_dir = None
    fit_output_dir = None
    fit_total_path = None
    template_candidate_paths = []

    if template_chain_paths:
        if protein_seq_path is None:
            progress("Protein sequence is unavailable, skip template fix/imp")
        elif final_denovo is None or (not os.path.exists(final_denovo)):
            progress("De novo protein model is unavailable, skip template fix/imp")
        else:
            protein_chain_dir = pjoin(temp_dir, "template_refine", "denovo_protein_chains")
            denovo_protein_chain_paths = _split_structure_to_chains(
                final_denovo,
                protein_chain_dir,
                protein_only=True,
                suffix="pdb",
            )
            if not denovo_protein_chain_paths:
                progress("No protein chains were found in the de novo model, skip template fix/imp")
            else:
                from em3dfold.template.pipeline import fix_pipeline, imp_pipeline
                from em3dfold.template.pipeline.template_refine import build_template_refine_context

                shared_context_dir = pjoin(temp_dir, "template_refine", "shared_context")
                os.makedirs(shared_context_dir, exist_ok=True)
                shared_context = build_template_refine_context(
                    chain_paths=denovo_protein_chain_paths,
                    template_paths=template_chain_paths,
                    lib_dir=script_dir,
                    work_dir=shared_context_dir,
                    seq_path=protein_seq_path,
                    verbose=False,
                    debug=False,
                    prepare_domains_flag=True,
                )
                print(f"# Shared protein template count = {len(shared_context.templates)}")

                if len(shared_context.templates) == 0:
                    progress("No valid protein templates remain after filtering, skip template fix/imp")
                else:
                    if not args.skip_fix:
                        progress_stage("fix")
                        start = time.time()
                        fix_output_dir = pjoin(temp_dir, "fix")
                        fix_args = argparse.Namespace(
                            seq=protein_seq_path,
                            chain=denovo_protein_chain_paths,
                            template=template_chain_paths,
                            lib=script_dir,
                            output=fix_output_dir,
                            verbose=False,
                            debug=False,
                        )
                        fix_pipeline.run_with_context(fix_args, shared_context)
                        end = time.time()
                        print("# Time = {:.4f}".format(end - start))
                    else:
                        progress("Skip template fix")

                    if not args.skip_imp:
                        progress_stage("imp")
                        start = time.time()
                        imp_output_dir = pjoin(temp_dir, "imp")
                        imp_args = argparse.Namespace(
                            seq=protein_seq_path,
                            chain=denovo_protein_chain_paths,
                            template=template_chain_paths,
                            lib=script_dir,
                            output=imp_output_dir,
                            verbose=False,
                            debug=False,
                        )
                        imp_pipeline.run_with_context(imp_args, shared_context)
                        end = time.time()
                        print("# Time = {:.4f}".format(end - start))
                    else:
                        progress("Skip template imp")
    elif args.protein_template:
        progress("No protein template chains were extracted, skip template fix/imp/fit")

    if template_chain_paths and (not args.skip_fit):
        fit_map_path = _first_existing_path(
            pjoin(temp_dir, "pred", "mc.mrc"),
            pjoin(temp_dir, "format_map.mrc"),
            args.map,
        )
        if fit_map_path is None or (not os.path.exists(fit_map_path)):
            raise FileNotFoundError("Cannot find a density map for template domain fitting.")

        progress_stage("fit")
        start = time.time()
        from em3dfold.template.pipeline import fit_pipeline as template_fit

        fit_output_dir = pjoin(temp_dir, "fit")
        fit_args = argparse.Namespace()
        fit_args.protein_template = template_chain_paths
        fit_args.map = fit_map_path
        fit_args.output = fit_output_dir
        fit_args.resolution = 6.0
        fit_args.threshold = 15.0
        fit_args.threshold_ratio = 0.10
        fit_args.rshift = 1.0
        fit_args.rmerge = 1.0
        fit_args.rmsdcut1 = 2.5
        fit_args.rmsdcut2 = 5.0
        fit_args.rigid_nleast = 5
        fit_args.rigid_cutoff_score_early = -1.5
        fit_args.rigid_cutoff_score_late = -0.5
        fit_args.rigid_skip_short_residues = 50
        fit_args.device = args.device
        fit_args.angle_step = 18.0
        fit_args.fgrid = 3.0
        fit_args.sgrid = 2.0
        fit_args.ntrans = 8
        fit_args.ntop = 10
        fit_summary = template_fit.main(fit_args)
        fit_total_path = fit_summary.get("fitted_total_path")
        end = time.time()
        print("# Time = {:.4f}".format(end - start))
    elif template_chain_paths:
        progress("Skip protein-template rigid fitting")

    if imp_output_dir is not None:
        imp_trimmed = pjoin(imp_output_dir, "imp_chains_trimmed.cif")
        if os.path.exists(imp_trimmed):
            template_candidate_paths.append(imp_trimmed)
    if fix_output_dir is not None:
        fix_chains = pjoin(fix_output_dir, "fix_chains_templs.cif")
        if os.path.exists(fix_chains):
            template_candidate_paths.append(fix_chains)
    if fit_total_path is not None and os.path.exists(fit_total_path):
        template_candidate_paths.append(fit_total_path)

    assemble_output_dir = None
    assembled_output_path = None
    has_template_candidates = len(template_candidate_paths) > 0
    assemble_candidate_paths = list(template_candidate_paths)
    if (not has_template_candidates) and final_denovo is not None and os.path.exists(final_denovo):
        assemble_candidate_paths.append(final_denovo)
    elif run_nucleic_input and final_denovo is not None and os.path.exists(final_denovo):
        assemble_candidate_paths.append(final_denovo)

    if template_chain_paths and (not args.skip_assemble):
        ca_map_path = _first_existing_path(pjoin(temp_dir, "pred", "ca.mrc"))
        if assemble_candidate_paths and ca_map_path is not None and os.path.exists(ca_map_path):
            progress_stage("assemble")
            start = time.time()
            from em3dfold.pipeline import assemble as chain_assemble

            assemble_output_dir = pjoin(temp_dir, "assemble")
            assemble_args = argparse.Namespace(
                structure_paths=assemble_candidate_paths,
                ca_map_path=ca_map_path,
                output=assemble_output_dir,
                clash_threshold=0.10,
                clash_distance=1.0,
                clash_resolution=5.0,
                map_percentile=99.9,
                time_limit=120.0,
                num_workers=4,
                log_search_progress=False,
            )
            chain_assemble.main(assemble_args)
            assembled_output_path = pjoin(assemble_output_dir, "assemble.cif")
            end = time.time()
            print("# Time = {:.4f}".format(end - start))
        elif assemble_candidate_paths:
            progress("CA map is unavailable, skip assemble")
        else:
            progress("No structures are available for assemble")
    elif template_chain_paths:
        progress("Skip assemble")
    else:
        progress("No protein template input, skip template fix/imp/fit/assemble")

    final_output_path = None
    for candidate in [
        assembled_output_path,
        final_denovo,
        fit_total_path,
        pjoin(imp_output_dir, "imp_chains_trimmed.cif") if imp_output_dir else None,
        pjoin(fix_output_dir, "fix_chains_templs.cif") if fix_output_dir else None,
    ]:
        if candidate is not None and os.path.exists(candidate):
            final_output_path = candidate
            break

    has_output = False
    if final_output_path is not None and os.path.exists(final_output_path):
        shutil.copy(final_output_path, fo)
        fix_quotes(fo)
        has_output = True


    # Remove temp files
    if not args.keep_temp_files:
        progress("Remove temporary files")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        else:
            pass
    else:
        progress("Keep temporary files")

    if has_output:
        progress("EM3DFold has completed the structure modeling process.")
        progress("Final model: {}".format(fo))
        if os.path.exists(fo_entropy):
            progress("Residue-type confidence file: {}".format(fo_entropy))
        if fix_output_dir is not None and os.path.exists(fix_output_dir):
            progress("Template fix results: {}".format(fix_output_dir))
        if imp_output_dir is not None and os.path.exists(imp_output_dir):
            progress("Template imp results: {}".format(imp_output_dir))
        if fit_output_dir is not None and os.path.exists(fit_output_dir):
            progress("Template fit results: {}".format(fit_output_dir))
        if assemble_output_dir is not None and os.path.exists(assemble_output_dir):
            progress("Assembled selection results: {}".format(assemble_output_dir))
        if args.keep_temp_files:
            progress("Intermediate files kept in: {}".format(temp_dir))
        progress("Done")
    else:
        progress("EM3DFold did not produce a final model.")
        progress("Please check run.log for the stage that failed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
