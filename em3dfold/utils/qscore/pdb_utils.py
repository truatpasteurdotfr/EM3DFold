from dataclasses import dataclass

import numpy as np

from em3dfold.io.pdbio import read_pdb


@dataclass(frozen=False)
class Protein:
    atom_positions: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    chain_index: np.ndarray


def get_protein_from_file_path(file_path: str, chain_id: str = None) -> Protein:
    if chain_id is not None:
        raise NotImplementedError("chain_id filtering is not implemented in EM3DFold qscore.")

    atom_positions, atom_mask, aatype, residue_index, chain_index = read_pdb(file_path, return_bfactor=False)
    return Protein(
        atom_positions=np.asarray(atom_positions, dtype=np.float32),
        aatype=np.asarray(aatype, dtype=np.int32),
        atom_mask=np.asarray(atom_mask, dtype=np.float32),
        residue_index=np.asarray(residue_index, dtype=np.int32),
        chain_index=np.asarray(chain_index, dtype=np.int32),
    )

