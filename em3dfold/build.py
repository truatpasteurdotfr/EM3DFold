"""Main program"""
import os
import sys
import time
import shutil
import argparse
import tempfile
from pathlib import Path

from em3dfold.io.pdbio import fix_quotes
from em3dfold.io.seqio import read_fasta
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
    parser.add_argument("--protein-chain", "-pc", help="Input protein chain template")
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
    print("# Start modeling")

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
        print("# Skip preprocess")


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
        print("# Skip dl cx")


    # ms
    if not args.skip_denovo:
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
            print("# Skip map to p")


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
                print("# Get protein LM embedding")
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
                print("# Skip protein LM embedding")

            if run_na:
                print("# Get nucleic-acid LM embedding")
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
                print("# Skip nucleic-acid LM embedding")

            initial_polymer_path = pjoin(temp_dir, "pred", "raw_polymer.pdb")
            _merge_initial_polymer_files(
                initial_polymer_path,
                [
                    (pjoin(temp_dir, "pred", "raw_ca.pdb") if run_protein else None, "A"),
                    (pjoin(temp_dir, "pred", "raw_c4.pdb") if run_na else None, "B"),
                ],
            )

            print("# Infer complex with a single inferlm_v3x run")
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
            print("# Skip unified inferlm_v3x")

    else:
        print("# Skip denovo modeling")

    final_denovo = _first_existing_path(
        pjoin(temp_dir, "denovo", "output.cif"),
        pjoin(temp_dir, "denovo", "denovo.cif"),
    )
    final_entropy = _first_existing_path(
        pjoin(temp_dir, "denovo", "output_entropy_score.cif"),
    )
    fo = pjoin(out_dir, "output.cif")
    fo_entropy = pjoin(out_dir, "output_entropy_score.cif")

    has_output = False
    if final_denovo is not None and os.path.exists(final_denovo):
        shutil.copy(final_denovo, fo)
        has_output = True
        fix_quotes(fo)
    if final_entropy is not None and os.path.exists(final_entropy):
        shutil.copy(final_entropy, fo_entropy)
        fix_quotes(fo_entropy)


    # Remove temp files
    if not args.keep_temp_files:
        print("# No keep temp files")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        else:
            pass
    else:
        print("# Keep temp files")

    if has_output:
        print("#" + " " + "-" * 70)
        print("# EM3DFold has completed the structure modeling process.")
        print("#" + " " + "-" * 70)
        print("# You can find the final model at: {}".format(fo))
        if os.path.exists(fo_entropy):
            print("# The residue-type confidence file is at: {}".format(fo_entropy))
        if args.keep_temp_files:
            print("# Intermediate files are kept in: {}".format(temp_dir))
        print("#" + " " + "-" * 70)
        print("# Done!")
    else:
        print("#" + " " + "-" * 70)
        print("# EM3DFold did not produce a final model.")
        print("#" + " " + "-" * 70)
        print("# Please check the logs above for the stage that failed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
