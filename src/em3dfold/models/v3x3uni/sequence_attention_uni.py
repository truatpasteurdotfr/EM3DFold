import math
from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from em3dfold.models.v3.sequence_attention import (
    get_batched_sequence_attention_features,
    get_batched_sequence_attention_scores,
)


class SequenceAttentionUni(nn.Module):
    """CryoAtom2-style unified sequence attention.

    Protein and nucleic-acid sequence embeddings are packed internally into a
    single feature stream, while a two-channel visibility mask controls which
    token type each residue node may attend to.
    """

    def __init__(
        self,
        d: int = 256,
        d_seq: int = 1280,
        d_seq_na: int | None = None,
        d_head: int = 48,
        n_head: int = 8,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.d = d
        self.d_seq_prot = d_seq
        self.d_seq_na = d_seq if d_seq_na is None else d_seq_na
        self.d_seq_packed = self.d_seq_prot + self.d_seq_na
        self.d_head = d_head
        self.n_head = n_head

        self.attention_scale = math.sqrt(self.d_head)
        self.norm = nn.LayerNorm(self.d)

        self.q = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=True),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.k = nn.Sequential(
            nn.Linear(self.d_seq_packed, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.v = nn.Sequential(
            nn.Linear(self.d_seq_packed, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=True),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
            nn.Sigmoid(),
        )
        self.back = nn.Sequential(
            Rearrange("... h d -> ... (h d)", h=self.n_head, d=self.d_head),
            nn.Linear(self.n_head * self.d_head, self.d, bias=False),
        )

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def _prepare_sequence_inputs(
        self,
        seq_emb,
        seq_mask,
        d_seq,
        batch_size,
        device,
        dtype,
    ):
        if seq_emb is None:
            seq_emb = torch.zeros(batch_size, 1, d_seq, device=device, dtype=dtype)
            seq_mask = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
            return seq_emb, seq_mask

        seq_emb = seq_emb.to(device=device, dtype=dtype)
        if seq_mask is None:
            seq_mask = torch.ones(seq_emb.shape[:2], device=device, dtype=torch.bool)
        else:
            seq_mask = seq_mask.to(device=device, dtype=torch.bool)
        return seq_emb, seq_mask

    def _pack_sequence_inputs(
        self,
        prot_seq_emb,
        prot_seq_mask,
        na_seq_emb,
        na_seq_mask,
    ):
        batch_size, prot_len, _ = prot_seq_emb.shape
        _, na_len, _ = na_seq_emb.shape
        device = prot_seq_emb.device
        dtype = prot_seq_emb.dtype

        prot_packed = torch.zeros(
            batch_size,
            prot_len,
            self.d_seq_packed,
            device=device,
            dtype=dtype,
        )
        prot_packed[..., : self.d_seq_prot] = prot_seq_emb

        na_packed = torch.zeros(
            batch_size,
            na_len,
            self.d_seq_packed,
            device=device,
            dtype=dtype,
        )
        na_packed[..., self.d_seq_prot :] = na_seq_emb

        packed_sequence_emb = torch.cat([prot_packed, na_packed], dim=1)

        prot_mask_2c = torch.zeros(batch_size, prot_len, 2, device=device, dtype=dtype)
        prot_mask_2c[..., 0] = prot_seq_mask.to(dtype=dtype)

        na_mask_2c = torch.zeros(batch_size, na_len, 2, device=device, dtype=dtype)
        na_mask_2c[..., 1] = na_seq_mask.to(dtype=dtype)

        packed_sequence_mask = torch.cat([prot_mask_2c, na_mask_2c], dim=1)
        return packed_sequence_emb, packed_sequence_mask

    def _build_visible_mask(self, prot_mask, batch, packed_sequence_mask):
        node_idx = torch.arange(prot_mask.shape[0], device=prot_mask.device)
        token_channel = torch.where(
            prot_mask.bool(),
            torch.zeros_like(prot_mask, dtype=torch.long),
            torch.ones_like(prot_mask, dtype=torch.long),
        )
        batched_mask = packed_sequence_mask[batch]
        return batched_mask[node_idx, :, token_channel]

    def forward_normal(
        self,
        node,
        prot_mask=None,
        batch=None,
        attention_batch_size=200,
        prot_seq_emb=None,
        prot_seq_mask=None,
        na_seq_emb=None,
        na_seq_mask=None,
        **kwargs,
    ):
        return self._intern_forward(
            node=node,
            prot_mask=prot_mask,
            batch=batch,
            attention_batch_size=attention_batch_size,
            prot_seq_emb=prot_seq_emb,
            prot_seq_mask=prot_seq_mask,
            na_seq_emb=na_seq_emb,
            na_seq_mask=na_seq_mask,
        )

    def forward_checkpoint(
        self,
        node,
        prot_mask=None,
        batch=None,
        attention_batch_size=200,
        prot_seq_emb=None,
        prot_seq_mask=None,
        na_seq_emb=None,
        na_seq_mask=None,
        **kwargs,
    ):
        if (not torch.is_grad_enabled()) or (not node.requires_grad):
            return self._intern_forward(
                node=node,
                prot_mask=prot_mask,
                batch=batch,
                attention_batch_size=attention_batch_size,
                prot_seq_emb=prot_seq_emb,
                prot_seq_mask=prot_seq_mask,
                na_seq_emb=na_seq_emb,
                na_seq_mask=na_seq_mask,
            )

        new_forward = partial(
            self._intern_forward,
            prot_mask=prot_mask,
            batch=batch,
            attention_batch_size=attention_batch_size,
            prot_seq_emb=prot_seq_emb,
            prot_seq_mask=prot_seq_mask,
            na_seq_emb=na_seq_emb,
            na_seq_mask=na_seq_mask,
        )
        return torch.utils.checkpoint.checkpoint(
            new_forward,
            node,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def _intern_forward(
        self,
        node,
        prot_mask,
        batch,
        attention_batch_size,
        prot_seq_emb,
        prot_seq_mask,
        na_seq_emb,
        na_seq_mask,
    ):
        device = node.device
        dtype = node.dtype

        if prot_mask is None:
            raise ValueError("prot_mask must be provided for unified sequence attention.")

        if batch is None:
            batch = torch.zeros(node.shape[0], dtype=torch.long, device=device)
        else:
            batch = batch.to(device=device, dtype=torch.long)

        prot_mask = prot_mask.to(device=device, dtype=torch.bool)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        prot_seq_emb, prot_seq_mask = self._prepare_sequence_inputs(
            prot_seq_emb,
            prot_seq_mask,
            self.d_seq_prot,
            batch_size,
            device,
            dtype,
        )
        na_seq_emb, na_seq_mask = self._prepare_sequence_inputs(
            na_seq_emb,
            na_seq_mask,
            self.d_seq_na,
            batch_size,
            device,
            dtype,
        )

        packed_sequence_emb, packed_sequence_mask = self._pack_sequence_inputs(
            prot_seq_emb,
            prot_seq_mask,
            na_seq_emb,
            na_seq_mask,
        )

        y = self.norm(node)
        sequence_query = self.q(y)
        sequence_key = self.k(packed_sequence_emb)
        sequence_value = self.v(packed_sequence_emb)

        sequence_attention_scores = get_batched_sequence_attention_scores(
            sequence_query,
            sequence_key,
            batch,
            self.attention_scale,
            batch_size=attention_batch_size,
            device=device,
        )

        visible_mask = self._build_visible_mask(
            prot_mask=prot_mask,
            batch=batch,
            packed_sequence_mask=packed_sequence_mask,
        )
        masked_scores = sequence_attention_scores.masked_fill(
            visible_mask[..., None] <= 0,
            -1e4,
        )
        sequence_attention_weights = torch.softmax(masked_scores, dim=1)

        new_features_attention = get_batched_sequence_attention_features(
            sequence_attention_weights,
            sequence_value,
            batch,
            batch_size=attention_batch_size,
            device=device,
        )

        gate = self.gate(y)
        new_features = self.back(gate * new_features_attention)
        return self.norm(math.sqrt(2) * node + new_features)
