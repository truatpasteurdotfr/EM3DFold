import torch
import numpy as np
from scipy.spatial import cKDTree

from em3dfold.polymer_utils import residue_constants as rc
from em3dfold.polymer_utils import polymer

def affines_and_torsion_angles_to_atomc_pos(
    pred_affines,
    pred_torsions,
    aatype=None,
):
    if isinstance(pred_affines, np.ndarray):
        pred_affines = torch.from_numpy(pred_affines)

    if isinstance(pred_torsions, np.ndarray):
        pred_torsions = torch.from_numpy(pred_torsions)

    # to all atom
    num_res = len(pred_affines)

    # assumes all protein
    if aatype is None:
        aatype = np.zeros(num_res, dtype=np.int32)

    all_frames = polymer.torsion_angles_to_frames(
        aatype,  # (N)
        pred_affines, # (N, 3, 4)
        pred_torsions, # (N, 10, 2)
    )

    atomc_positions, atomc_mask = polymer.frames_and_literature_positions_to_atomc_pos(
        aatype,
        all_frames,
        return_mask=True,
    )

    return atomc_positions, atomc_mask

