import torch.nn as nn

from em3dfold.models.v3x3.model import Model as V3X3Model
from em3dfold.models.v3x3uni.sequence_attention_uni import SequenceAttentionUni


class Model(V3X3Model):
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
            pred_ss=pred_ss,
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
