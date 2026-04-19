import os
import time
import torch
import random
import argparse
import numpy as np
from math import ceil
from torch import FloatTensor as FT
import warnings

warnings.filterwarnings("ignore")

from em3dfold.scunet.scunet import SCUNet as Model
from em3dfold.utils.torch_utils import get_device_names
from em3dfold.utils.cryo_utils import (
    parse_map,
    write_map,
    pad_map,
    chunk_generator,
    get_batch_from_generator,
    map_batch_to_map,
)
from em3dfold.utils.log_utils import progress, progress_stage
from em3dfold.utils.misc_utils import pjoin, abspath
from em3dfold.utils.torch_utils import clear_cuda_cache

EM_WEIGHTS_ENV_VAR = "EM_WEIGHTS_DIR"
PROGRESS_LOGGER_NAME = "em3dfold.pred.progress"


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _resolve_env_weights_root():
    env_value = os.environ.get(EM_WEIGHTS_ENV_VAR)
    if env_value is None or str(env_value).strip() == "":
        return None
    return os.path.realpath(os.path.expanduser(env_value))


def _resolve_model_dir(model_dir, dir_script):
    if model_dir is not None:
        return abspath(model_dir)

    env_root = _resolve_env_weights_root()
    if env_root is not None:
        candidate_weights_dir = os.path.join(env_root, "weights")
        if os.path.exists(candidate_weights_dir):
            return candidate_weights_dir
        if os.path.exists(env_root):
            return env_root
        return candidate_weights_dir

    return pjoin(dir_script, "..", "weights")


def load_model_and_run_inference_on_map(model_file, map_file, **kwargs):
    apix = kwargs["apix"]
    stride = kwargs["stride"]
    box_size = kwargs["box_size"]
    n_classes = kwargs["n_classes"]
    batch_size = kwargs["batch_size"]
    device = kwargs["device"]
    scale = kwargs["scale"]

    print(f"# Load map data from {map_file}")
    em_map, origin, nxyz, voxel_size = parse_map(map_file, ignorestart=False, apix=apix)

    print(f"# Map dimensions = {nxyz}")
    if np.max(nxyz) > 400:
        print("# Current map is a bit large, the run may be a little slow. Please wait patiently.")

    if np.min(nxyz) > 400:
        stride = max(stride, 16)

    model_state_dict = torch.load(model_file, map_location="cpu")
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

    model = Model(input_resolution=box_size, n_classes=n_classes)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    padded_map = pad_map(em_map, box_size, dtype=np.float32, padding=0.0)
    positive_values = em_map[em_map > 0]
    if positive_values.size > 0:
        maximum = np.percentile(positive_values, 99.999)
    else:
        maximum = float(np.max(em_map))

    map_pred = np.zeros((n_classes,) + padded_map.shape, dtype=np.float32)
    denominator = np.zeros((n_classes,) + padded_map.shape, dtype=np.float32)

    print("# Start processing")
    generator = chunk_generator(padded_map, maximum, box_size, stride)
    ncx, ncy, ncz = [ceil(nxyz[2 - i] / stride) for i in range(3)]
    total_steps = float(ncx * ncy * ncz)
    acc_steps, acc_steps_x, l_bar = 0.0, 0, 0

    ts = time.time()
    with torch.inference_mode():
        while True:
            positions, chunks = get_batch_from_generator(generator, batch_size, dtype=np.float32)

            chunks /= scale

            if len(positions) == 0:
                break

            acc_steps += len(chunks)
            acc_steps_x = int((acc_steps / total_steps) * 100.0) // 5
            if acc_steps_x > l_bar:
                l_bar = acc_steps_x
                te = time.time()
                bar = f"|{'#' * (2 * l_bar)}{'-' * ((20 - l_bar) * 2)}| {int(l_bar * 5)}% {te - ts:.4f} seconds elapsed"
                print(f"\r{bar}", flush=True)

            x_batch = FT(chunks).view(-1, 1, box_size, box_size, box_size).to(device)
            y_pred = model(x_batch)
            y_pred = y_pred.cpu().detach().numpy()
            map_pred, denominator = map_batch_to_map(map_pred, denominator, positions, y_pred, box_size)

    map_pred = (map_pred / denominator.clip(min=1))[
        :,
        box_size : box_size + nxyz[2],
        box_size : box_size + nxyz[1],
        box_size : box_size + nxyz[0],
    ]

    if acc_steps < total_steps:
        print("\r|########################################| 100%", flush=True)

    del model
    del model_state_dict
    clear_cuda_cache(device, note=os.path.basename(model_file))
    return map_pred, em_map, origin, nxyz, voxel_size


data_params = {
    "apix": 1.0,
    "box_size": 48,
    "stride": 16,
}

test_params = {
    "batch_size": 160,
}


def inference_segmentation(dir_map, contour, dir_model, dir_out, data_params, test_params, device):
    print(f"# Select map contour at {contour:.6f}", flush=True)
    print(f"# Running on device {device}")

    map_pred, em_map, origin, _, voxel_size = load_model_and_run_inference_on_map(
        model_file=dir_model,
        map_file=dir_map,
        apix=data_params["apix"],
        box_size=data_params["box_size"],
        stride=24, # hard coded, larger stride for time saving
        n_classes=3,
        batch_size=test_params["batch_size"],
        device=device,
        scale=1.0, # 100.0
    )

    map_pred = np.argmax(map_pred, axis=0)
    below_contour = np.where(em_map <= contour, 3, 0)
    map_pred = np.where(below_contour > map_pred, below_contour, map_pred)
    types = ["prot.mrc", "na.mrc", "bg.mrc"]

    for i in [0, 1]:
        mask = np.where(map_pred == i, 1, 0)
        out = mask * em_map
        dir_map_out = os.path.join(dir_out, types[i])
        write_map(dir_map_out, out.astype(np.float32), voxel_size, origin=origin)
        print(f"# Write map to {dir_map_out}", flush=True)

    n_prot = np.where(map_pred == 0, 1, 0).sum()
    n_na = np.where(map_pred == 1, 1, 0).sum()
    n_bg = np.where(map_pred == 2, 1, 0).sum()
    denom_all = n_prot + n_na + n_bg + 1e-3
    denom_mix = n_prot + n_na + 1e-3

    print("# Region report")
    print(f"# Among all voxels. prot ratio is {n_prot / denom_all:.4f}")
    print(f"# Among all voxels. na   ratio is {n_na / denom_all:.4f}")
    print(f"# Among all voxels. bg   ratio is {n_bg / denom_all:.4f}")
    print(f"# Among prot and na voxels. prot ratio is {n_prot / denom_mix:.4f}")
    print(f"# Among prot and na voxels. na   ratio is {n_na / denom_mix:.4f}", flush=True)


def inference_nucleic_c4(dir_map, contour, dir_model, dir_out, data_params, test_params, device):
    print(f"# Select map contour at {contour:.6f}", flush=True)
    print(f"# Running on device {device}")

    map_pred, em_map, origin, _, voxel_size = load_model_and_run_inference_on_map(
        model_file=dir_model,
        map_file=dir_map,
        apix=data_params["apix"],
        box_size=data_params["box_size"],
        stride=16, # 12
        n_classes=1,
        batch_size=test_params["batch_size"],
        device=device,
        scale=1.0,
    )
    mask = np.where(em_map <= contour, 0, 1).astype(np.int8)
    out = mask * map_pred[0]
    dir_map_out = pjoin(dir_out, "c4.mrc")
    write_map(dir_map_out, out.astype(np.float32), voxel_size, origin=origin)
    print(f"# Write map to {dir_map_out}", flush=True)


def inference_nucleic_aa(dir_map, dir_model, dir_out, data_params, test_params, device):
    print(f"# Running on device {device}")

    map_pred, _, origin, _, voxel_size = load_model_and_run_inference_on_map(
        model_file=dir_model,
        map_file=dir_map,
        apix=data_params["apix"],
        box_size=data_params["box_size"],
        stride=16, # 12
        n_classes=4,
        batch_size=test_params["batch_size"],
        device=device,
        scale=1.0,
    )

    map_pred[[1, 2]] = map_pred[[2, 1]]
    dir_map_out = pjoin(dir_out, "logits.npz")
    np.savez(dir_map_out, map=map_pred.astype(np.float32), origin=origin, voxel_size=voxel_size)
    print(f"# Write aa logits file to {dir_map_out}", flush=True)


def inference_protein_ca(dir_map, contour, dir_model, dir_out, data_params, test_params, device):
    print(f"# Select map contour at {contour:.6f}", flush=True)
    print(f"# Running on device {device}")

    map_pred, em_map, origin, _, voxel_size = load_model_and_run_inference_on_map(
        model_file=dir_model,
        map_file=dir_map,
        apix=data_params["apix"],
        box_size=data_params["box_size"],
        stride=data_params["stride"],
        n_classes=3,
        batch_size=test_params["batch_size"],
        device=device,
        scale=1.0,
    )

    mask = np.where(em_map <= contour, 0, 1).astype(np.int8)
    ca_out = mask * map_pred[1]
    ca_map_out = os.path.join(dir_out, "ca.mrc")
    write_map(ca_map_out, ca_out.astype(np.float32), voxel_size, origin=origin)
    print(f"# Write map to {ca_map_out}", flush=True)

    backbone_out = mask * np.mean(map_pred[:3], axis=0)
    backbone_map_out = os.path.join(dir_out, "mc.mrc")
    write_map(backbone_map_out, backbone_out.astype(np.float32), voxel_size, origin=origin)
    print(f"# Write map to {backbone_map_out}", flush=True)


def main(args):
    seed_torch(42)

    cpu_num = 4
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    start = time.time()
    dir_script = abspath(os.path.dirname(__file__))
    print(f"# Script dir is {dir_script}")

    dir_map = abspath(args.input)
    dir_out = abspath(args.output)
    contour = args.contour
    dir_model = _resolve_model_dir(args.model, dir_script)
    print(f"# Specify model path to {dir_model}")

    if not args.protein and not args.nucleic:
        raise Exception("Please specify at least one prediction target: --protein and/or --nucleic")

    if args.protein and not os.path.exists(pjoin(dir_model, "protein", "model_prot_mc")):
        raise Exception("Cannot find model weights for protein CA")
    if args.nucleic and not os.path.exists(pjoin(dir_model, "na", "model_na_c4")):
        raise Exception("Cannot find model weights for na C4")
    if args.nucleic and not os.path.exists(pjoin(dir_model, "na", "model_na_aa")):
        raise Exception("Cannot find model weights for na AA")
    if not os.path.exists(pjoin(dir_model, "cpx", "model_seg")):
        raise Exception("Cannot find model weights for cpx seg")

    print(f"# Making directory {dir_out}", flush=True)
    os.makedirs(dir_out, exist_ok=True)
    progress(f"Write logs and outputs to {dir_out}", logger_name=PROGRESS_LOGGER_NAME)

    if isinstance(args.stride, int):
        assert 12 <= args.stride <= 48, f"Invalid stride = {args.stride} -> 12 <= stride <= 48"
        data_params["stride"] = args.stride

    if args.batchsize is not None:
        test_params["batch_size"] = args.batchsize

    devices = get_device_names(args.device)
    device = devices[0]

    progress_stage("Start segmentation", logger_name=PROGRESS_LOGGER_NAME)
    inference_segmentation(
        dir_map=dir_map,
        contour=contour,
        dir_model=pjoin(dir_model, "cpx", "model_seg"),
        dir_out=dir_out,
        data_params=data_params,
        test_params=test_params,
        device=device,
    )
    progress("Done segmentation", logger_name=PROGRESS_LOGGER_NAME)

    if args.protein:
        progress_stage("Start protein CA prediction", logger_name=PROGRESS_LOGGER_NAME)
        inference_protein_ca(
            dir_map=pjoin(dir_out, "prot.mrc"),
            contour=contour,
            dir_model=pjoin(dir_model, "protein", "model_prot_mc"),
            dir_out=dir_out,
            data_params=data_params,
            test_params=test_params,
            device=device,
        )
        progress("Done protein CA prediction", logger_name=PROGRESS_LOGGER_NAME)

    if args.nucleic:
        progress_stage("Start nucleic C4' prediction", logger_name=PROGRESS_LOGGER_NAME)
        inference_nucleic_c4(
            dir_map=pjoin(dir_out, "na.mrc"),
            contour=contour,
            dir_model=pjoin(dir_model, "na", "model_na_c4"),
            dir_out=dir_out,
            data_params=data_params,
            test_params=test_params,
            device=device,
        )
        progress("Done nucleic C4' prediction", logger_name=PROGRESS_LOGGER_NAME)

        progress_stage("Start nucleic AA prediction", logger_name=PROGRESS_LOGGER_NAME)
        inference_nucleic_aa(
            dir_map=pjoin(dir_out, "na.mrc"),
            dir_model=pjoin(dir_model, "na", "model_na_aa"),
            dir_out=dir_out,
            data_params=data_params,
            test_params=test_params,
            device=device,
        )
        progress("Done nucleic AA prediction", logger_name=PROGRESS_LOGGER_NAME)

    clear_cuda_cache(device, note="pred")

    end = time.time()
    print(f"# Time consuming {end - start:.4f}", flush=True)
    progress(f"Prediction completed in {end - start:.2f}s", logger_name=PROGRESS_LOGGER_NAME)


def add_args(parser):
    script_dir = abspath(os.path.dirname(__file__))
    parser.add_argument("--input", "-i", type=str, required=True, help="Input EM density map file")
    parser.add_argument("--output", "-o", type=str, default="./", help="Output directory of predicted maps")
    parser.add_argument("--contour", "-c", type=float, default=1e-6, help="Input contour level")
    parser.add_argument("--batchsize", "-b", type=int, default=40, help="Batchsize for prediction")
    parser.add_argument("--device", "-g", type=str, help="Which GPU to use, '0' for #0", default="0")
    parser.add_argument(
        "--model",
        "--weights-dir",
        "-m",
        type=str,
        dest="model",
        help="Directory to deep learning models (expects subdirs: na/, protein/)",
        default=None,
    )
    parser.add_argument("--stride", "-s", type=int, help="Stride for splitting chunks", default=16)
    parser.add_argument("--protein", action="store_true", help="Predict protein CA map (ca.mrc) and backbone map (backbone.mrc)")
    parser.add_argument("--nucleic", action="store_true", help="Predict nucleic maps (c4.mrc and logits.npz)")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = add_args(parser).parse_args()
    main(args)
