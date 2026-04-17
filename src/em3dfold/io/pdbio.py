import re
import os
import math
from typing import List
from copy import deepcopy
from collections import defaultdict, OrderedDict

import numpy as np
import torch

from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO, mmcif_order
from Bio.PDB import PDBParser, MMCIFParser

from em3dfold.polymer_utils import residue_constants as rc

# Read chain names at local
with open( os.path.join(os.path.dirname(__file__), "chain_names.txt" ), "r") as f:
    chain_names = f.readline()
    chain_names = chain_names.strip().split("|")

class CIFXIO(MMCIFIO):
    def _save_dict(self, out_file):
        label_seq_id = deepcopy(self.dic["_atom_site.auth_seq_id"])
        auth_seq_id = deepcopy(self.dic["_atom_site.auth_seq_id"])
        self.dic["_atom_site.label_seq_id"] = label_seq_id
        self.dic["_atom_site.auth_seq_id"] = auth_seq_id

        # Adding missing "pdbx_formal_charge", "auth_comp_id", "auth_atom_id" to complete a record
        N = len(self.dic["_atom_site.group_PDB"])
        self.dic["_atom_site.pdbx_formal_charge"] = ["?"]*N
        self.dic["_atom_site.auth_comp_id"] = deepcopy(self.dic["_atom_site.label_comp_id"])
        self.dic["_atom_site.auth_asym_id"] = deepcopy(self.dic["_atom_site.label_asym_id"])
        self.dic["_atom_site.auth_atom_id"] = deepcopy(self.dic["_atom_site.label_atom_id"])

        # Handle an extra space at the end of _atom_site.xxx
        _atom_site = mmcif_order["_atom_site"]
        _atom_site = [x.strip() + " " for x in _atom_site]
        mmcif_order["_atom_site"] = _atom_site

        new_dic = defaultdict()
        for k, v in self.dic.items():
            if k[:11] == "_atom_site.":
                new_k = k.strip() + " "
            else:
                new_k = k
            new_dic[new_k] = v
        self.dic = new_dic

        return super()._save_dict(out_file)

def split_to_chains(chain_idx, *args):
    # split
    n_chain = chain_idx.max() + 1
    ret = ()
    for t in args:
        list_t = []
        for i in range(n_chain):
            mask = chain_idx == i
            list_t.append(t[mask])

        ret += (list_t, )
        
    return ret


def convert_to_chains(chain_idxs, *inputs):
    return split_to_chains(chain_idxs, *inputs)

def fix_quotes(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        # Handle '' issues
        fixed_lines = [re.sub(r"'([A-Z0-9]+)''", '"\\1\'"', line) for line in lines]
        temp_filename = filename + ".tmp"
        with open(temp_filename, 'w') as f:
            f.writelines(fixed_lines)
        os.replace(temp_filename, filename)
    except IOError as e:
        print(f"Error processing file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def chains_atom_pos_to_pdb(
    filename,
    chains_atom_pos,
    chains_atom_mask,
    chains_res_type,
    chains_res_idx=None,
    chains_idx=None, 
    chains_bfactor=None,
    chains_occupancy=None, 
    suffix='cif',
    remarks=None,
):
    # For different chains
    assert len(chains_atom_pos) == len(chains_atom_mask)
    if chains_occupancy is None:
        chains_occupancy = []
        for k in range(len(chains_atom_pos)):
            chains_occupancy.append( np.full(len(chains_atom_pos[k]), 1.0, dtype=np.float32) )

    if chains_bfactor is None:
        chains_bfactor = []
        for k in range(len(chains_atom_pos)):
            chains_bfactor.append( np.ones_like(chains_atom_mask[k], dtype=np.float32) * 0.0 )

    if chains_res_type is None:
        chains_res_type = []
        for k in range(len(chains_atom_pos)):
            chains_res_type.append( np.full(len(chains_atom_pos[k]), 0, dtype=np.int32) )

    if chains_res_idx is None:
        chains_res_idx = []
        for k in range(len(chains_atom_pos)):
            chains_res_idx.append( np.arange(len(chains_atom_pos[k]), dtype=np.int32) )

    if chains_idx is None:
        chains_idx = []
        for k in range(len(chains_atom_pos)):
            chains_idx.append( k )

    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")

    n_total_atom = 0
    for k in range(len(chains_atom_pos)):
        # For each chain
        chain_atom_pos = chains_atom_pos[k]
        chain_atom_mask = chains_atom_mask[k]

        chain_res_type = chains_res_type[k]
        chain_res_idx = chains_res_idx[k]
        chain_idx = chains_idx[k]
        chain_bfactor = chains_bfactor[k]

        # Init a new chain
        struct.init_chain(chain_names[chain_idx])

        for i in range(len(chain_atom_pos)):
            # For each residue
            res_type = chain_res_type[i]
            res_idx = chain_res_idx[i]

            # [0, 20) for protein, [20, 28) for NA
            if not (0 <= res_type < 28):
                continue

            res_name_3 = rc.index_to_restype_3[res_type]
            atom_names = rc.restype3_to_atoms[res_name_3]

            n_atom = len(chain_atom_pos[i])
            if len(atom_names) > n_atom:
                atom_names = atom_names[:n_atom]

            field_name = " "

            struct.init_residue(res_name_3, field_name, res_idx, " ")

            # atom14 or atom23 is both OK
            for atom_name, pos, bfactor, mask in zip(
                atom_names, chain_atom_pos[i], chain_bfactor[i], chain_atom_mask[i]
            ):

                if atom_name is None or \
                   atom_name == "" or \
                   mask < 1 or \
                   np.any(np.isnan( pos )) or \
                   np.all(np.abs(pos) < 1e-3):
                    continue

                struct.set_line_counter(n_total_atom + 1)

                struct.init_atom(
                    name=atom_name,
                    coord=pos,
                    b_factor=bfactor,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=atom_name,
                    element=atom_name[0],
                )
                n_total_atom += 1

    struct = struct.get_structure()
    if suffix in ['cif', '.cif', 'CIF', '.CIF', 'mmcif', 'MMCIF']:
        io = CIFXIO()
        io.set_structure(struct)
        io.save(filename)

        # Fix quotes
        fix_quotes(filename)
    else:
        io = PDBIO()
        io.set_structure(struct)
        io.save(filename, write_end=False)

# read pdb
def read_pdb(
    filename, 
    ignore_hetatm=False,
    keep_valid=False,
    return_bfactor=False,
):
    # Read file
    if filename.endswith("pdb"):
        parser = PDBParser()
    elif filename.endswith("cif"):
        parser = MMCIFParser()
    else:
        raise Exception("Error only support pdb/cif file")
    structure = parser.get_structure('pdb', filename)

    # Only use model 0
    model = structure[0]

    # Extract residue information
    atom_pos = []
    atom_mask = []
    res_type = []
    res_idx = []
    chain_idx = []
    bfactor = []

    valid_resname_3 = set(rc.index_to_restype_3[:28])
    for i, chain in enumerate(model):
        prev_residue_number = None
        for residue_index, residue in enumerate(chain):
            residue_number = residue.get_id()[1]
            if prev_residue_number is None or residue_number != prev_residue_number:
                resname_3 = residue.get_resname().strip()
                hetfield, resseq, icode = residue.get_id()

                # If hetatom
                if ignore_hetatm and hetfield != " ":
                    continue

                # If unknown residues
                if resname_3 not in valid_resname_3:
                    continue

                curr_res_type = rc.restype_3_to_index[resname_3]
                atom_names = rc.restype3_to_atoms[resname_3]

                prev_residue_number = residue_number

            coords = []
            mask = []
            bf = []

            # Get atom types for current residue
            while len(atom_names) < 23:
                atom_names.append("")

            for atom_index in atom_names:
                try:
                    atom = residue[atom_index]
                    coords.append(atom.get_coord())
                    mask.append(1)
                    bf.append(atom.get_bfactor())
                except KeyError:
                    coords.append([float("nan") for _ in range(3)])
                    mask.append(0)
                    bf.append(float("nan"))

            atom_pos.append(coords)
            atom_mask.append(mask)
            chain_idx.append(i)
            res_type.append(curr_res_type)
            res_idx.append(residue_number)

            bfactor.append(bf)

    # Convert to NumPy arrays
    atom_pos = np.array(atom_pos).astype(np.float32)
    atom_mask = np.array(atom_mask).astype(np.int32)
    res_type = np.array(res_type).astype(np.int32)
    res_idx = np.array(res_idx).astype(np.int32)
    chain_idx = np.array(chain_idx).astype(np.int32)
    bfactor = np.array(bfactor).astype(np.float32)

    chain_idx = chain_idx - chain_idx.min()

    if keep_valid:
        idxs = np.all(atom_mask[:, :3], axis=-1)
        if len(idxs) == 0 or not np.any(idxs):
            raise Exception("# Error cannot find any amino-acid in the pdb")
        atom_pos = atom_pos[idxs]
        atom_mask = atom_mask[idxs]
        res_type = res_type[idxs]
        res_idx = res_idx[idxs]
        chain_idx = chain_idx[idxs]
        bfactor = bfactor[idxs]

    if return_bfactor:
        return atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor
    return atom_pos, atom_mask, res_type, res_idx, chain_idx



def chains_atom_pos_to_pdb_bb_simple(
    filename,
    chains_atom_pos,
    chains_atom_mask=None,
    chains_res_type=None,
    chains_res_idx=None,
):
    assert filename.endswith(".pdb")
    n = 0
    with open(filename, 'w') as f:
        for chain_atom_pos in chains_atom_pos:
            for k in range(len(chain_atom_pos)):
                f.write("ATOM  {:>5d}  N   ALA A{:>4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(

                    n + 1,
                    n + 1,
                    chain_atom_pos[k, 0, 0],
                    chain_atom_pos[k, 0, 1],
                    chain_atom_pos[k, 0, 2],
                ))
                f.write("ATOM  {:>5d}  CA  ALA A{:>4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                    n + 1,
                    n + 1,
                    chain_atom_pos[k, 1, 0],
                    chain_atom_pos[k, 1, 1],
                    chain_atom_pos[k, 1, 2],
                ))
                f.write("ATOM  {:>5d}  C   ALA A{:>4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                    n + 1,
                    n + 1,
                    chain_atom_pos[k, 1, 0],
                    chain_atom_pos[k, 1, 1],
                    chain_atom_pos[k, 1, 2],
                ))
                f.write("TER\n")


def split_atoms_to_chains(atom_pos, chain_idx):
    n = np.max(chain_idx) + 1
    chains = []
    for i in range(n):
        idx = chain_idx == i
        chains.append(atom_pos[idx])
    return chains


def write_points_as_pdb(filename, res_pos, res_types, bfactors=None, ter=False, chain='A', atom=" CA "):
    assert len(res_types) >= len(res_pos)
    with open(filename, 'w', encoding='utf-8') as f:
        if bfactors is None:
            bfactors = np.ones(len(res_pos))
        for i in range(len(res_pos)):
            n = i + 1
            f.write("ATOM  {:5d} {:4s} {:>3s}{:>2s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
                n, atom, res_types[i], chain, n, res_pos[i][0], res_pos[i][1], res_pos[i][2], bfactors[i], bfactors[i]
            ))
            if ter:
                f.write("TER\n")
        if not ter:
            f.write("TER\n")


def chains_to_pdb(filename, chains, res_name="ALA", atom_name=" CA "):
    with open(filename, 'w', encoding='utf-8') as f:
        n = 0
        for i, chain in enumerate(chains):
            chain_id = str(i) if i < 100 else 99
            for k in range(len(chain)):
                f.write("ATOM  {:5d} {:4s} {:>3s}{:>2s}{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(
                    n, atom_name, res_name, chain_id, n, chain[k][0], chain[k][1], chain[k][2]
                ))
                n += 1
            f.write("TER\n")


def chains_atom_pos_to_cif(
        filename, 
        chains_atom_pos, 
        chains_atom_mask, 
        chains_res_type, 
        chains_res_idx, 
        chains_bfactor=None, 
        chains_occupancy=None,
        suffix="cif",
    ):
    return chains_atom_pos_to_pdb(
        filename,
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_type=chains_res_type,
        chains_res_idx=chains_res_idx,
        chains_bfactor=chains_bfactor,
        chains_occupancy=chains_occupancy,
        suffix=suffix, 
    )


def atom3_to_atom14(atom3_pos):
    raise NotImplementedError


def ca_to_atom3(ca):
    l = ca.shape[0]
    atom3_pos = np.zeros((l, 3, 3), dtype=np.float32)
    atom3_mask = np.zeros((l, 3), dtype=np.int32)
    for i in range(l):
        atom3_pos[i][1] = ca[i]
        atom3_mask[i][1] = 1
    return atom3_pos, atom3_mask


def write_atoms_as_pdb(filename, atom_pos, res_type=None, bfactor=None, ter=True, final_ter=False):
    restypes_3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]

    if atom_pos.ndim == 2 and atom_pos.shape[-1] == 3:
        atom_names = [" CA "]
        atom_pos = atom_pos[:, None, :]
    elif atom_pos.ndim == 3 and atom_pos.shape[-1] == 3 and atom_pos.shape[1] == 1:
        atom_names = [" CA "]
    elif atom_pos.ndim == 3 and atom_pos.shape[-1] == 3 and atom_pos.shape[1] >= 3:
        atom_names = [" N  ", " CA ", " C  "]
        atom_pos = atom_pos[:, :3, :]
    else:
        raise Exception(f"# Error shape - {atom_pos.shape}")

    if np.any(np.isnan(atom_pos)):
        raise Exception("# Error atom pos have NaNs")

    if res_type is None:
        res_type = np.zeros(len(atom_pos), dtype=np.int32)
    if bfactor is None:
        bfactor = np.ones(len(atom_pos), dtype=np.float32) * 100.0

    n_res = 1
    n_atom = 1
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(atom_pos)):
            residue = atom_pos[i]
            residue_type = restypes_3[res_type[i]]
            for k in range(len(residue)):
                atom_name = atom_names[k]
                f.write(
                    "ATOM  {:>5d} {:4s} {:3s}{:>2s}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}\n".format(
                        n_atom if n_atom <= 99999 else 99999,
                        atom_name,
                        residue_type,
                        "A",
                        n_res if n_res <= 9999 else 9999,
                        residue[k][0],
                        residue[k][1],
                        residue[k][2],
                        1.0,
                        bfactor[i],
                    )
                )
                n_atom += 1
            n_res += 1
            if ter:
                f.write("TER\n")
        if final_ter:
            f.write("TER\n")

