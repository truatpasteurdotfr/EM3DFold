import math
from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from em3dfold.utils.torch_utils import get_batch_slices, padded_sequence_softmax


def get_batched_sequence_attention_scores(
    sequence_query,
    sequence_key,
    batch,
    attention_scale,
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_query.shape[0],
        sequence_key.shape[1],
        sequence_key.shape[2],
        device=device,
    )

    seq_batches = get_batch_slices(output.shape[1], batch_size)
    sequence_query = sequence_query[:, None]

    for seq_batch in seq_batches:
        output[:, seq_batch] = (
            sequence_query * sequence_key[:, seq_batch][batch]
        ).sum(dim=-1) / attention_scale

    return output


def get_batched_sequence_attention_features(
    sequence_attention_weights,
    sequence_value,
    batch,
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_attention_weights.shape[0],
        sequence_value.shape[2],
        sequence_value.shape[3],
        device=device,
    )
    seq_batches = get_batch_slices(sequence_attention_weights.shape[1], batch_size)

    for seq_batch in seq_batches:
        output += (
            sequence_attention_weights[:, seq_batch][..., None]
            * sequence_value[:, seq_batch][batch]
        ).sum(dim=1)
    return output


class SequenceAttention(nn.Module):
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
        self.d_head = d_head
        self.n_head = n_head

        self.attention_scale = math.sqrt(self.d_head)
        self.norm = nn.LayerNorm(self.d)

        self.q_prot = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.q_na = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )

        self.k_prot = nn.Sequential(
            nn.Linear(self.d_seq_prot, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.v_prot = nn.Sequential(
            nn.Linear(self.d_seq_prot, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.k_na = nn.Sequential(
            nn.Linear(self.d_seq_na, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )
        self.v_na = nn.Sequential(
            nn.Linear(self.d_seq_na, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
        )

        self.gate_prot = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
            nn.Sigmoid(),
        )
        self.gate_na = nn.Sequential(
            nn.Linear(self.d, self.n_head * self.d_head, bias=False),
            Rearrange("... (h d) -> ... h d", h=self.n_head, d=self.d_head),
            nn.Sigmoid(),
        )

        self.back_prot = nn.Sequential(
            Rearrange("... h d -> ... (h d)", h=self.n_head, d=self.d_head),
            nn.Linear(self.n_head * self.d_head, self.d, bias=False),
        )
        self.back_na = nn.Sequential(
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
            seq_emb = torch.zeros(batch_size, 3, d_seq, device=device, dtype=dtype)
            seq_mask = torch.ones(batch_size, 3, device=device, dtype=torch.bool)
            return seq_emb, seq_mask

        if seq_mask is None:
            seq_mask = torch.ones(seq_emb.shape[:2], device=seq_emb.device, dtype=torch.bool)

        return seq_emb, seq_mask

    def _attend_subset(
        self,
        node_subset,
        batch_subset,
        sequence_emb,
        sequence_mask,
        attention_batch_size,
        device,
        q_module,
        k_module,
        v_module,
        gate_module,
        back_module,
    ):
        y = self.norm(node_subset)
        sequence_query = q_module(y)
        sequence_key = k_module(sequence_emb)
        sequence_value = v_module(sequence_emb)

        sequence_attention_scores = get_batched_sequence_attention_scores(
            sequence_query,
            sequence_key,
            batch_subset,
            self.attention_scale,
            batch_size=attention_batch_size,
            device=device,
        )

        batched_mask = sequence_mask[batch_subset].unsqueeze(-1)
        sequence_attention_weights = padded_sequence_softmax(
            sequence_attention_scores, batched_mask, dim=1
        )

        new_features_attention = get_batched_sequence_attention_features(
            sequence_attention_weights,
            sequence_value,
            batch_subset,
            batch_size=attention_batch_size,
            device=device,
        )

        gate = gate_module(y)
        new_features = back_module(gate * new_features_attention)
        return new_features

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
        if prot_mask is None:
            raise ValueError("prot_mask must be provided for mixed protein/NA sequence attention.")

        if batch is None:
            batch = torch.zeros(node.shape[0], dtype=torch.long, device=device)

        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        dtype = node.dtype

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

        prot_indices = torch.nonzero(prot_mask.bool(), as_tuple=False).flatten()
        na_indices = torch.nonzero(~prot_mask.bool(), as_tuple=False).flatten()

        output = node.clone()
        if prot_indices.numel() > 0:
            output[prot_indices] = output[prot_indices] + self._attend_subset(
                node_subset=node[prot_indices],
                batch_subset=batch[prot_indices],
                sequence_emb=prot_seq_emb,
                sequence_mask=prot_seq_mask,
                attention_batch_size=attention_batch_size,
                device=device,
                q_module=self.q_prot,
                k_module=self.k_prot,
                v_module=self.v_prot,
                gate_module=self.gate_prot,
                back_module=self.back_prot,
            )
        if na_indices.numel() > 0:
            output[na_indices] = output[na_indices] + self._attend_subset(
                node_subset=node[na_indices],
                batch_subset=batch[na_indices],
                sequence_emb=na_seq_emb,
                sequence_mask=na_seq_mask,
                attention_batch_size=attention_batch_size,
                device=device,
                q_module=self.q_na,
                k_module=self.k_na,
                v_module=self.v_na,
                gate_module=self.gate_na,
                back_module=self.back_na,
            )
        return output
