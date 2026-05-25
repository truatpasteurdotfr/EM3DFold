import contextlib

import torch
import torch.nn as nn

from em3dfold.models.v3x3uni.sequence_attention_uni import SequenceAttentionUni
from em3dfold.models.v3x4.model import Model as V3X4Model
from em3dfold.models.v2.model_refresh import Output


class Model(V3X4Model):
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
        rotation_stop_gradient=False,
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
            pred_ss=pred_ss,
            pred_na_type=pred_na_type,
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
            edge_bias_rbf_bins=edge_bias_rbf_bins,
        )
        self.rotation_stop_gradient = rotation_stop_gradient

        if self.use_seq_attn:
            self.seq_attn_blocks = nn.ModuleList(
                [
                    SequenceAttentionUni(
                        d=d_node,
                        d_seq=d_seq,
                        d_seq_na=self.d_seq_na,
                        d_head=d_head,
                        n_head=n_head,
                        checkpoint=use_checkpoint,
                    )
                    for _ in range(self.n_block // self.seq_attn_every)
                ]
            )

    def _maybe_stop_rotation_gradient(self, affines):
        if not self.rotation_stop_gradient:
            return affines
        rot = affines[..., :3, :3].detach()
        trans = affines[..., :3, 3:]
        return torch.cat([rot, trans], dim=-1)

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
                cryo_init_affines = self._maybe_stop_rotation_gradient(init_affines)
                node_density, edge_density, bde_out = self._run_cryo_init(
                    affines=cryo_init_affines,
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
                edge_bias_affines = self._maybe_stop_rotation_gradient(init_affines)
                edge_bias = self._compute_edge_bias(
                    edge_bias_affines,
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
                        refresh_affines = self._maybe_stop_rotation_gradient(affines)
                        node_density, edge_density, refresh_bde_out = self._run_cryo_init(
                            affines=refresh_affines,
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
                        edge_bias_affines = self._maybe_stop_rotation_gradient(affines)
                        edge_bias = self._compute_edge_bias(
                            edge_bias_affines,
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
