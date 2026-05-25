import contextlib

import torch
import torch.nn as nn

from em3dfold.models.v2.model_refresh import rbf
from em3dfold.models.v2.model_refresh import Output
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
        pred_na_type=False,
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

        self.pred_na_type = pred_na_type
        if self.pred_na_type:
            self.na_type_predictor = nn.Sequential(
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, d_node),
                nn.ReLU(),
                nn.Linear(d_node, 2),
            )

        self.edge_bias_rbf_bins = edge_bias_rbf_bins

        proxy_prot_atom_names = rc.restype_name_to_atomc_names["ALA"]
        proxy_na_atom_names = rc.restype_name_to_atomc_names["A"]
        self.proxy_slot_atom_names = [
            "N",
            "CA",
            "C",
            "O",
            "CB",
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

        prot_present_atom_names = {"N", "CA", "C", "O", "CB"}
        na_present_atom_names = {
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
        }

        def _build_proxy_indices_and_mask(atom_names, present_atom_names):
            present_indices = [atom_names.index(atom_name) for atom_name in present_atom_names]
            fallback_index = present_indices[0]
            slot_indices = []
            slot_mask = []
            for slot_atom_name in self.proxy_slot_atom_names:
                if slot_atom_name in present_atom_names:
                    slot_indices.append(atom_names.index(slot_atom_name))
                    slot_mask.append(1.0)
                else:
                    slot_indices.append(fallback_index)
                    slot_mask.append(0.0)
            return slot_indices, slot_mask

        prot_proxy_atom_indices, prot_proxy_atom_mask = _build_proxy_indices_and_mask(
            proxy_prot_atom_names,
            prot_present_atom_names,
        )
        na_proxy_atom_indices, na_proxy_atom_mask = _build_proxy_indices_and_mask(
            proxy_na_atom_names,
            na_present_atom_names,
        )

        self.register_buffer(
            "proxy_atom_indices",
            torch.tensor(
                [
                    prot_proxy_atom_indices,
                    na_proxy_atom_indices,
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "proxy_atom_mask",
            torch.tensor(
                [
                    prot_proxy_atom_mask,
                    na_proxy_atom_mask,
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.num_proxy_atoms = len(self.proxy_slot_atom_names)

        fallback_slot_names = [
            ["N", "CA", "C"],
            ["C3'", "C4'", "O4'"],
        ]
        proxy_restype_names = ["ALA", "A"]
        fallback_slot_positions = []
        fallback_slot_masks = []
        for type_idx, restype_name in enumerate(proxy_restype_names):
            atom_names = proxy_prot_atom_names if type_idx == 0 else proxy_na_atom_names
            restype = rc.restype_3_to_index[restype_name]
            slot_positions = []
            slot_mask = []
            fallback_slot_name_set = set(fallback_slot_names[type_idx])
            for slot_atom_name in self.proxy_slot_atom_names:
                if slot_atom_name in fallback_slot_name_set:
                    atom_index = atom_names.index(slot_atom_name)
                    slot_positions.append(
                        torch.tensor(
                            rc.restype_atomc_rigid_group_positions[restype, atom_index],
                            dtype=torch.float32,
                        )
                    )
                    slot_mask.append(1.0)
                else:
                    slot_positions.append(torch.zeros(3, dtype=torch.float32))
                    slot_mask.append(0.0)
            fallback_slot_positions.append(torch.stack(slot_positions, dim=0))
            fallback_slot_masks.append(torch.tensor(slot_mask, dtype=torch.float32))

        self.register_buffer(
            "fallback_local_positions",
            torch.stack(fallback_slot_positions, dim=0),
            persistent=False,
        )
        self.register_buffer(
            "fallback_atom_mask",
            torch.stack(fallback_slot_masks, dim=0),
            persistent=False,
        )

        self.edge_bias_embed = nn.Sequential(
            nn.Linear(
                self.num_proxy_atoms * self.num_proxy_atoms * self.edge_bias_rbf_bins,
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
                atom_mask_selector = self.fallback_atom_mask[prot_mask.long()].to(
                    device=device,
                    dtype=affines.dtype,
                )
                fallback_local_positions = self.fallback_local_positions[prot_mask.long()].to(
                    device=device,
                    dtype=affines.dtype,
                )
                anchor_atoms = affine_mul_vecs(
                    affines[:, None, :, :],
                    fallback_local_positions,
                )
            else:
                atom_mask_selector = self.proxy_atom_mask[prot_mask.long()].to(
                    device=device,
                    dtype=affines.dtype,
                )
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
                atom_index_selector = self.proxy_atom_indices[prot_mask.long()].to(device=device)
                gather_index = atom_index_selector[..., None].expand(-1, -1, 3)
                anchor_atoms = torch.gather(proxy_atomc_positions, dim=1, index=gather_index)

            neighbor_atom_mask = atom_mask_selector[edge_index]
            atom_pair_mask = atom_mask_selector[:, None, :, None] * neighbor_atom_mask[:, :, None, :]
            neighbor_atoms = anchor_atoms[edge_index]
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

        return self.edge_bias_embed(atom_pair_rbf)

    def _merge_edge_bias(self, edge, edge_bias):
        return edge + edge_bias

    def forward(
        self,
        affines,
        prot_mask,
        cryo_grids=None,
        cryo_global_origins=None,
        cryo_voxel_sizes=None,
        prot_seq_embed=None,
        prot_seq_embed_mask=None,
        na_seq_embed=None,
        na_seq_embed_mask=None,
        batch=None,
        run_iters=1,
        prev_node=None,
        prev_aa_probs=None,
        prev_rmsd=None,
    ):
        if batch is None:
            batch = torch.zeros(len(affines), device=affines.device, dtype=torch.long)

        if cryo_grids is None or cryo_global_origins is None or cryo_voxel_sizes is None:
            raise ValueError("Cryo grid inputs must be provided.")

        init_affines = affines
        init_node = torch.zeros((len(affines), self.d_node), device=affines.device)
        max_k = min(self.k, len(affines) - 1)
        init_edge = torch.zeros((len(affines), max_k, self.d_edge), device=affines.device)
        init_aa_logits = torch.zeros(
            (len(affines), self.num_res_types),
            device=affines.device,
        )
        init_edge_aa_logits = torch.zeros(
            (len(affines), max_k, self.num_res_types * self.num_res_types),
            device=affines.device,
        )
        init_rmsd = torch.zeros(
            (len(affines), 1),
            device=affines.device,
        )
        if prev_node is not None:
            if prev_node.shape != init_node.shape:
                raise ValueError(
                    f"prev_node must have shape {tuple(init_node.shape)}, "
                    f"got {tuple(prev_node.shape)}"
                )
            init_node = prev_node.to(
                device=affines.device,
                dtype=init_node.dtype,
            )
        if prev_aa_probs is not None:
            if prev_aa_probs.shape != init_aa_logits.shape:
                raise ValueError(
                    f"prev_aa_probs must have shape {tuple(init_aa_logits.shape)}, "
                    f"got {tuple(prev_aa_probs.shape)}"
                )
            init_aa_logits = prev_aa_probs.to(
                device=affines.device,
                dtype=init_aa_logits.dtype,
            )
        if prev_rmsd is not None:
            if prev_rmsd.shape == (len(affines),):
                prev_rmsd = prev_rmsd[..., None]
            if prev_rmsd.shape != init_rmsd.shape:
                raise ValueError(
                    f"prev_rmsd must have shape {tuple(init_rmsd.shape)}, "
                    f"got {tuple(prev_rmsd.shape)}"
                )
            init_rmsd = prev_rmsd.to(
                device=affines.device,
                dtype=init_rmsd.dtype,
            )

        fixed_edge_index = None
        fixed_full_edge_index = None

        pred_aatype = None
        pred_edge_aa_logits = None
        pred_node_existence = None
        pred_edge_existence = None
        pred_pairing = None

        for run_iter in range(run_iters):
            with torch.no_grad() if run_iter < run_iters - 1 else contextlib.nullcontext():
                node_density, edge_density, bde_out = self._run_cryo_init(
                    affines=init_affines,
                    cryo_grids=cryo_grids,
                    cryo_global_origins=cryo_global_origins,
                    cryo_voxel_sizes=cryo_voxel_sizes,
                    edge_index=fixed_edge_index,
                    full_edge_index=fixed_full_edge_index,
                    batch=batch,
                )
                if fixed_edge_index is None:
                    fixed_edge_index = bde_out.edge_index
                    fixed_full_edge_index = bde_out.full_edge_index

                node = node_density
                edge = edge_density
                node_density_state = node_density
                edge_density_state = edge_density

                node_type, edge_type = self.embed_mol_type(prot_mask, edge_index=bde_out.edge_index)
                node = node + node_type
                edge = edge + edge_type

                node = node + self.embed_cycle_node(init_node)
                edge = edge + self.embed_cycle_edge(init_edge)
                node = node + self.embed_cycle_aa(init_aa_logits)
                edge = edge + self.embed_cycle_edge_aa(init_edge_aa_logits)
                node_conf_feat, edge_conf_feat = self._build_confidence_features(
                    init_rmsd,
                    bde_out.edge_index,
                )
                node = node + self.embed_cycle_conf(node_conf_feat)
                edge = edge + self.embed_cycle_edge_conf(edge_conf_feat)

                torsion_angles = None
                edge_bias = self._compute_edge_bias(
                    init_affines,
                    torsion_angles,
                    prot_mask,
                    bde_out.edge_index,
                )
                edge = self._merge_edge_bias(edge, edge_bias)

                affines_list = []
                torsion_list = []
                affines = init_affines
                pos_emb = bde_out.pos3d_emb
                seq_attn_idx = 0
                for i in range(self.n_block):
                    do_seq_attn = (
                        self.use_seq_attn
                        and self.seq_attn_every > 0
                        and ((i + 1) % self.seq_attn_every == 0)
                    )
                    do_geometry_update = self.blocks[i].enable_geometry_update

                    if do_seq_attn:
                        node = self.seq_attn_blocks[seq_attn_idx](
                            node,
                            prot_mask=prot_mask.bool(),
                            batch=batch,
                            prot_seq_emb=prot_seq_embed,
                            prot_seq_mask=prot_seq_embed_mask,
                            na_seq_emb=na_seq_embed,
                            na_seq_mask=na_seq_embed_mask,
                        )
                        seq_attn_idx += 1

                    node, edge, affines, torsion_angles = self.blocks[i](
                        node,
                        edge,
                        affines,
                        pos_emb=pos_emb,
                        edge_index=bde_out.edge_index,
                        use_checkpoint=self.use_checkpoint,
                    )

                    if do_geometry_update and i != self.n_block - 1:
                        node_density, edge_density, refresh_bde_out = self._run_cryo_init(
                            affines=affines,
                            cryo_grids=cryo_grids,
                            cryo_global_origins=cryo_global_origins,
                            cryo_voxel_sizes=cryo_voxel_sizes,
                            edge_index=fixed_edge_index,
                            full_edge_index=fixed_full_edge_index,
                            batch=batch,
                        )
                        if not self.has_intermediate_refresh:
                            raise RuntimeError(
                                "Intermediate density refresh triggered without refresh modules."
                            )
                        node_density_state = node_density_state + self.refresh_node_head_embed(node_density)
                        edge_density_state = edge_density_state + self.refresh_edge_head_embed(edge_density)
                        node = node + self.refresh_node_embed(node_density)
                        edge = edge + self.refresh_edge_embed(edge_density)
                        edge_bias = self._compute_edge_bias(
                            affines,
                            torsion_angles,
                            prot_mask,
                            bde_out.edge_index,
                        )
                        edge = self._merge_edge_bias(edge, edge_bias)
                        pos_emb = refresh_bde_out.pos3d_emb

                    if do_geometry_update:
                        affines_list.append(affines)
                        torsion_list.append(torsion_angles)

                pred_aatype = self.aa_predictor(node_density_state, node)
                pred_edge_aa_logits = self.edge_aa_predictor(edge_density_state, edge)
                rmsd = self.rmsd_predictor(node)

                if self.pred_node_exist:
                    pred_node_existence = self.node_existence_predictor(node)

                if self.pred_edge_exist:
                    pred_edge_existence = self.edge_existence_predictor(edge)

                if self.pred_pairing:
                    pred_pairing = self.pairing_predictor(edge)

                init_affines = affines
                init_node = node
                init_edge = edge

                with torch.no_grad():
                    prot_probs = torch.softmax(pred_aatype[..., :20], dim=-1)
                    na_probs = torch.softmax(pred_aatype[..., 20:], dim=-1)
                    init_aa_logits = torch.zeros_like(pred_aatype)
                    prot_mask_bool = prot_mask.bool().unsqueeze(-1)
                    init_aa_logits[..., :20] = torch.where(
                        prot_mask_bool,
                        prot_probs,
                        torch.zeros_like(prot_probs),
                    )
                    init_aa_logits[..., 20:] = torch.where(
                        ~prot_mask_bool,
                        na_probs,
                        torch.zeros_like(na_probs),
                    )

                    init_edge_aa_logits = torch.softmax(
                        pred_edge_aa_logits.reshape(
                            pred_edge_aa_logits.shape[0],
                            pred_edge_aa_logits.shape[1],
                            -1,
                        ),
                        dim=-1,
                    )

                with torch.no_grad():
                    init_rmsd = rmsd

        output = Output(
            prot_mask=prot_mask,
            pred_aatype=pred_aatype,
            pred_aa_logits=pred_aatype,
            pred_edge_aa_logits=pred_edge_aa_logits,
            pred_torsions=torsion_list,
            pred_affines=affines_list,
            pred_positions=[x[..., :3, -1] for x in affines_list],
            pred_rmsd=rmsd,
            recycle_node_state=node,
            pred_node_existence=pred_node_existence if self.pred_node_exist else None,
            pred_edge_existence=pred_edge_existence if self.pred_edge_exist else None,
            edge_index=bde_out.edge_index,
            full_edge_index=bde_out.full_edge_index,
            pred_pairing=pred_pairing if self.pred_pairing else None,
        )
        if self.pred_ss:
            output["pred_ss_logits"] = self.ss_predictor(output["recycle_node_state"])
        else:
            output["pred_ss_logits"] = None
        if self.pred_na_type:
            output["pred_na_type_logits"] = self.na_type_predictor(output["recycle_node_state"])
        else:
            output["pred_na_type_logits"] = None
        return output
