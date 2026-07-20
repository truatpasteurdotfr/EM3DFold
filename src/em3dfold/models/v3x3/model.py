import torch
import torch.nn as nn

from em3dfold.models.v2.model_refresh import rbf
from em3dfold.models.v3x.model import Model as V3XModel
from em3dfold.polymer_utils import polymer
from em3dfold.polymer_utils import residue_constants as rc
from em3dfold.utils.affine_utils import affine_mul_vecs


class Model(V3XModel):
    def __init__(
        self,
        d_node=256,
        d_edge=256,
        d_head=48,
        d_seq=1280,
        d_seq_na=1280,
        n_qk_point=4,
        n_v_point=8,
        n_head=8,
        n_block=12,
        k=32,
        p_drop=0.10,
        c_grid=1,
        pred_node_exist=False,
        pred_edge_exist=True,
        pred_pairing=False,
        pred_ss=True,
        use_checkpoint=False,
        use_seq_attn=True,
        seq_attn_every=2,
        geometry_update_every=4,
        recycle_use_full_structure=True,
        d_cryo_emb=None,
        cube_size=23,
        rectangle_length=15,
        cube_scunet_head_dim=16,
        cube_scunet_window_size=3,
        cube_scunet_drop_path=0.0,
        cube_scunet_trans_ratio=0.25,
        cube_scunet_max_trans_dim=128,
        edge_bias_rbf_bins=24,
    ):
        super().__init__(
            d_node=d_node,
            d_edge=d_edge,
            d_head=d_head,
            d_seq=d_seq,
            d_seq_na=d_seq_na,
            n_qk_point=n_qk_point,
            n_v_point=n_v_point,
            n_head=n_head,
            n_block=n_block,
            k=k,
            p_drop=p_drop,
            c_grid=c_grid,
            pred_node_exist=pred_node_exist,
            pred_edge_exist=pred_edge_exist,
            pred_pairing=pred_pairing,
            use_checkpoint=use_checkpoint,
            use_seq_attn=use_seq_attn,
            seq_attn_every=seq_attn_every,
            geometry_update_every=geometry_update_every,
            recycle_use_full_structure=recycle_use_full_structure,
            d_cryo_emb=d_node if d_cryo_emb is None else d_cryo_emb,
            cube_size=cube_size,
            rectangle_length=rectangle_length,
            cube_scunet_head_dim=cube_scunet_head_dim,
            cube_scunet_window_size=cube_scunet_window_size,
            cube_scunet_drop_path=cube_scunet_drop_path,
            cube_scunet_trans_ratio=cube_scunet_trans_ratio,
            cube_scunet_max_trans_dim=cube_scunet_max_trans_dim,
        )

        self.pred_ss = pred_ss
        if self.pred_ss:
            self.ss_predictor = nn.Sequential(
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, 3),
            )
        self.edge_bias_rbf_bins = edge_bias_rbf_bins
        self.edge_bias_type_dim = 4

        proxy_prot_atom_names = rc.restype_name_to_atomc_names["ALA"]
        proxy_na_atom_names = rc.restype_name_to_atomc_names["A"]
        na_proxy_atom_names = [
            "P",
            "O5'",
            "C5'",
            "C4'",
            "O4'",
            "C3'",
            "O3'",
            "C2'",
            "C1'",
            "N9",
        ]
        prot_proxy_atom_indices = [
            proxy_prot_atom_names.index("N"),
            proxy_prot_atom_names.index("CA"),
            proxy_prot_atom_names.index("C"),
            proxy_prot_atom_names.index("O"),
            proxy_prot_atom_names.index("CB"),
        ]
        na_proxy_atom_indices = [
            proxy_na_atom_names.index(atom_name)
            for atom_name in na_proxy_atom_names
        ]
        max_proxy_atoms = max(len(prot_proxy_atom_indices), len(na_proxy_atom_indices))

        def _pad_indices(indices):
            return indices + [indices[-1]] * (max_proxy_atoms - len(indices))

        def _mask_for(indices):
            return [1.0] * len(indices) + [0.0] * (max_proxy_atoms - len(indices))

        self.register_buffer(
            "proxy_atom_indices",
            torch.tensor(
                [
                    _pad_indices(prot_proxy_atom_indices),
                    _pad_indices(na_proxy_atom_indices),
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "proxy_atom_mask",
            torch.tensor(
                [
                    _mask_for(prot_proxy_atom_indices),
                    _mask_for(na_proxy_atom_indices),
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.num_proxy_atoms = int(self.proxy_atom_indices.shape[-1])
        self.register_buffer(
            "fallback_proxy_atom_indices",
            torch.tensor(
                [
                    [
                        proxy_prot_atom_names.index("N"),
                        proxy_prot_atom_names.index("CA"),
                        proxy_prot_atom_names.index("C"),
                    ],
                    [
                        proxy_na_atom_names.index("C3'"),
                        proxy_na_atom_names.index("C4'"),
                        proxy_na_atom_names.index("O4'"),
                    ],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        proxy_restypes = torch.tensor(
            [
                rc.restype_3_to_index["ALA"],
                rc.restype_3_to_index["A"],
            ],
            dtype=torch.long,
        )
        fallback_local_positions = []
        for type_idx, restype in enumerate(proxy_restypes.tolist()):
            atom_indices = self.fallback_proxy_atom_indices[type_idx]
            fallback_local_positions.append(
                torch.tensor(
                    rc.restype_atomc_rigid_group_positions[restype, atom_indices],
                    dtype=torch.float32,
                )
            )
        self.register_buffer(
            "fallback_local_positions",
            torch.stack(fallback_local_positions, dim=0),
            persistent=False,
        )
        self.edge_bias_embed = nn.Sequential(
            nn.Linear(
                self.num_proxy_atoms * self.num_proxy_atoms * self.edge_bias_rbf_bins
                + self.edge_bias_type_dim,
                d_edge,
                bias=False,
            ),
        )

    def _compute_edge_bias(self, affines, torsion_angles, prot_mask, edge_index):
        with torch.no_grad():
            device = affines.device

            if not self.recycle_use_full_structure:
                torsion_angles = None

            if torsion_angles is None:
                fallback_local_positions = self.fallback_local_positions[prot_mask.long()].to(device=device, dtype=affines.dtype)
                fallback_anchor_atoms = affine_mul_vecs(
                    affines[:, None, :, :],
                    fallback_local_positions,
                )
                fallback_neighbor_atoms = fallback_anchor_atoms[edge_index]
                atom_pair_dist = torch.norm(
                    fallback_anchor_atoms[:, None, :, None, :] - fallback_neighbor_atoms[:, :, None, :, :],
                    dim=-1,
                )
                atom_pair_dist = atom_pair_dist.reshape(atom_pair_dist.shape[0], atom_pair_dist.shape[1], -1)
                atom_pair_rbf = rbf(
                    atom_pair_dist,
                    d_count=self.edge_bias_rbf_bins,
                )
                atom_pair_rbf = atom_pair_rbf.reshape(
                    atom_pair_rbf.shape[0],
                    atom_pair_rbf.shape[1],
                    -1,
                )
                if atom_pair_dist.shape[-1] != self.num_proxy_atoms * self.num_proxy_atoms:
                    pad_size = (self.num_proxy_atoms * self.num_proxy_atoms) - atom_pair_dist.shape[-1]
                    if pad_size < 0:
                        raise ValueError("Fallback atom-pair feature dimension exceeds configured proxy dimension.")
                    if pad_size > 0:
                        pad_rbf = torch.zeros(
                            atom_pair_rbf.shape[0],
                            atom_pair_rbf.shape[1],
                            pad_size * self.edge_bias_rbf_bins,
                            device=device,
                            dtype=atom_pair_rbf.dtype,
                        )
                        atom_pair_rbf = torch.cat([atom_pair_rbf, pad_rbf], dim=-1)
            else:
                proxy_aatype = torch.where(
                    prot_mask.bool(),
                    torch.zeros_like(prot_mask, dtype=torch.long),
                    torch.full_like(prot_mask, 20, dtype=torch.long),
                )
                proxy_torsion_angles = rc.select_torsion_angles(
                    torsion_angles,
                    proxy_aatype,
                    normalize=True,
                )
                proxy_frames = polymer.torsion_angles_to_frames(
                    proxy_aatype.detach().cpu().numpy(),
                    affines,
                    proxy_torsion_angles,
                )
                proxy_atomc_positions = polymer.frames_and_literature_positions_to_atomc_pos(
                    proxy_aatype.detach().cpu().numpy(),
                    proxy_frames,
                )

                atom_index_selector = self.proxy_atom_indices[prot_mask.long()].to(device)
                atom_mask_selector = self.proxy_atom_mask[prot_mask.long()].to(device=device, dtype=affines.dtype)
                gather_index = atom_index_selector[..., None].expand(-1, -1, 3)
                anchor_atoms = torch.gather(proxy_atomc_positions, dim=1, index=gather_index)
                neighbor_atoms = anchor_atoms[edge_index]
                neighbor_atom_mask = atom_mask_selector[edge_index]
                atom_pair_mask = atom_mask_selector[:, None, :, None] * neighbor_atom_mask[:, :, None, :]

                atom_pair_dist = torch.norm(
                    anchor_atoms[:, None, :, None, :] - neighbor_atoms[:, :, None, :, :],
                    dim=-1,
                )
                atom_pair_rbf = rbf(
                    atom_pair_dist.reshape(atom_pair_dist.shape[0], atom_pair_dist.shape[1], -1),
                    d_count=self.edge_bias_rbf_bins,
                )
                atom_pair_rbf = atom_pair_rbf * atom_pair_mask.reshape(
                    atom_pair_mask.shape[0],
                    atom_pair_mask.shape[1],
                    -1,
                    1,
                )
                atom_pair_rbf = atom_pair_rbf.reshape(
                    atom_pair_rbf.shape[0],
                    atom_pair_rbf.shape[1],
                    -1,
                )

            src_is_prot = prot_mask.bool()
            dst_is_prot = src_is_prot[edge_index]
            edge_type_feat = torch.stack(
                [
                    (~src_is_prot)[:, None] & (~dst_is_prot),
                    src_is_prot[:, None] & dst_is_prot,
                    (~src_is_prot)[:, None] & dst_is_prot,
                    src_is_prot[:, None] & (~dst_is_prot),
                ],
                dim=-1,
            ).to(dtype=atom_pair_rbf.dtype)
            atom_pair_rbf = torch.cat([atom_pair_rbf, edge_type_feat], dim=-1)

        return self.edge_bias_embed(atom_pair_rbf)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if self.pred_ss:
            output["pred_ss_logits"] = self.ss_predictor(output["recycle_node_state"])
        else:
            output["pred_ss_logits"] = None
        return output
