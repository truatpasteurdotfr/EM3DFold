from collections import namedtuple

import torch
from torch import nn

from em3dfold.models.knn_graph_torch import knn_graph
from em3dfold.models.modules import SinusoidalPositionalEncoding
from em3dfold.polymer_utils import residue_constants as rc


BackboneGeometryEmbeddingOutput = namedtuple(
    "BackboneGeometryEmbeddingOutput",
    [
        "position_embedding",
        "positions",
        "neighbor_vectors",
        "pair_geometry",
        "edge_index",
        "full_edge_index",
    ],
)


def _rbf(distances, d_count=20, d_min=2.0, d_max=32.0):
    device = distances.device
    centers = torch.linspace(d_min, d_max, d_count, device=device)
    centers = centers.view(1, 1, 1, 1, -1)
    sigma = (d_max - d_min) / d_count
    expanded = distances.unsqueeze(-1)
    return torch.exp(-((expanded - centers) / sigma) ** 2)


class BackboneGeometryEmbedding(nn.Module):
    def __init__(
        self,
        num_neighbors: int = 32,
        position_encoding_dim: int = 16,
        pair_geometry_dim: int = 128,
        num_rbf: int = 20,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.position_encoding_dim = position_encoding_dim
        self.pair_geometry_dim = pair_geometry_dim
        self.num_rbf = num_rbf

        self.position_encoding = SinusoidalPositionalEncoding(self.position_encoding_dim)
        self.proxy_atom_specs = {
            "protein": {
                "residue_name": "ALA",
                "atom_names": ["N", "CA", "C", "O"],
            },
            "na": {
                "residue_name": "DA",
                "atom_names": ["P", "C4'", "N1", "O3'"],
            },
        }
        self.num_protein_proxy_atoms = len(self.proxy_atom_specs["protein"]["atom_names"])
        self.num_na_proxy_atoms = len(self.proxy_atom_specs["na"]["atom_names"])
        self.num_pair_atoms = self.num_protein_proxy_atoms + self.num_na_proxy_atoms
        self.pair_geometry_projection = nn.Linear(
            self.num_pair_atoms * self.num_pair_atoms * self.num_rbf,
            self.pair_geometry_dim,
            bias=False,
        )

        self.proxy_atom_indices = {
            key: torch.tensor(
                [
                    rc.restype3_to_atoms[spec["residue_name"]].index(atom_name)
                    for atom_name in spec["atom_names"]
                ],
                dtype=torch.long,
            )
            for key, spec in self.proxy_atom_specs.items()
        }
        self.proxy_aatype = {
            key: rc.restype_3_to_index[spec["residue_name"]]
            for key, spec in self.proxy_atom_specs.items()
        }

    def _predict_proxy_atoms(self, affines, torsion_angles, proxy_key: str):
        from em3dfold.polymer_utils import polymer

        atom_indices = self.proxy_atom_indices[proxy_key].to(affines.device)
        residue_index = self.proxy_aatype[proxy_key]
        aatype = torch.full(
            (len(affines),),
            fill_value=residue_index,
            dtype=torch.long,
            device=affines.device,
        )
        residue_torsions = rc.select_torsion_angles(
            torsion_angles,
            aatype,
            normalize=True,
        )
        rigid_frames = polymer.torsion_angles_to_frames(
            aatype.detach().cpu().numpy(),
            affines,
            residue_torsions,
        )
        atom_positions = polymer.frames_and_literature_positions_to_atomc_pos(
            aatype.detach().cpu().numpy(),
            rigid_frames,
        )
        return atom_positions[:, atom_indices]

    def _build_pair_geometry(self, affines, edge_index, prot_mask, torsion_angles):
        if torsion_angles is None:
            zeros = torch.zeros(
                affines.shape[0],
                edge_index.shape[1],
                self.num_pair_atoms * self.num_pair_atoms * self.num_rbf,
                device=affines.device,
                dtype=affines.dtype,
            )
            return self.pair_geometry_projection(zeros)

        with torch.no_grad():
            prot_atoms = self._predict_proxy_atoms(affines, torsion_angles, "protein")
            na_atoms = self._predict_proxy_atoms(affines, torsion_angles, "na")
            node_atoms = torch.cat(
                [
                    prot_atoms,
                    na_atoms,
                ],
                dim=-2,
            )

            node_atom_mask = torch.zeros(
                node_atoms.shape[:-1],
                device=affines.device,
                dtype=affines.dtype,
            )
            node_atom_mask[prot_mask][..., self.num_protein_proxy_atoms :] = 1
            node_atom_mask[~prot_mask][..., : self.num_protein_proxy_atoms] = 1

            neighbor_mask = node_atom_mask.unsqueeze(1).unsqueeze(3) + node_atom_mask[edge_index].unsqueeze(2)
            pair_distances = (
                node_atoms.unsqueeze(1).unsqueeze(3) - node_atoms[edge_index].unsqueeze(2)
            ).norm(dim=-1)
            pair_distances = pair_distances + 1e3 * neighbor_mask
            pair_rbf = _rbf(pair_distances, d_count=self.num_rbf).flatten(2)

        return self.pair_geometry_projection(pair_rbf.requires_grad_())

    def forward(
        self,
        affines,
        prot_mask,
        torsion_angles=None,
        edge_index=None,
        full_edge_index=None,
        batch=None,
        k=None,
    ):
        positions = affines[..., :3, 3]
        num_neighbors = self.num_neighbors if k is None else k

        if (edge_index is None) != (full_edge_index is None):
            raise ValueError(
                "edge_index and full_edge_index must either both be None or both be provided."
            )

        if edge_index is None:
            full_edge_index = knn_graph(
                positions,
                num_neighbors,
                batch=batch,
                loop=False,
                flow="source_to_target",
            )
            edge_index = full_edge_index[0].reshape(len(positions), num_neighbors)

        position_embedding = self.position_encoding(positions).flatten(1)
        neighbor_vectors = positions[edge_index] - positions[:, None]
        pair_geometry = self._build_pair_geometry(
            affines=affines,
            edge_index=edge_index,
            prot_mask=prot_mask.bool(),
            torsion_angles=torsion_angles,
        )

        return BackboneGeometryEmbeddingOutput(
            position_embedding=position_embedding,
            positions=positions,
            neighbor_vectors=neighbor_vectors,
            pair_geometry=pair_geometry,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
        )
