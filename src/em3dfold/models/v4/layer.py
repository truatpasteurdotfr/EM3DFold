import math

import einops
import torch
from torch import nn
from einops.layers.torch import Rearrange

from em3dfold.models.modules import qk_modulation
from em3dfold.models.v4.sequence_attention import SequenceAttention


class GatedTransition(nn.Module):
    def __init__(self, in_features: int, expansion: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.gate_proj = nn.Linear(in_features, in_features * expansion, bias=False)
        self.value_proj = nn.Linear(in_features, in_features * expansion, bias=False)
        self.output_proj = nn.Linear(in_features * expansion, in_features, bias=False)

    def forward(self, x):
        y = torch.nn.functional.silu(self.gate_proj(x)) * self.value_proj(x)
        y = self.output_proj(y)
        return self.norm(y + math.sqrt(2) * x)


class NodeNeighborhoodAttention(nn.Module):
    def __init__(self, d_node: int, d_edge: int, n_head: int, d_head: int):
        super().__init__()
        self.scale = math.sqrt(d_head)
        self.node_norm = nn.LayerNorm(d_node)
        self.query_proj = nn.Sequential(
            nn.Linear(d_node, n_head * d_head, bias=False),
            Rearrange("n (h d) -> n h d", h=n_head, d=d_head),
        )
        self.key_proj = nn.Sequential(
            nn.Linear(d_node, n_head * d_head, bias=False),
            Rearrange("n k (h d) -> n k h d", h=n_head, d=d_head),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(d_node, n_head * d_head, bias=False),
            Rearrange("n k (h d) -> n k h d", h=n_head, d=d_head),
        )
        self.attention_bias = nn.Sequential(
            nn.LayerNorm(d_edge),
            nn.Linear(d_edge, n_head, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_node, n_head * d_head, bias=True),
            Rearrange("n (h d) -> n h d", h=n_head, d=d_head),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Sequential(
            Rearrange("n h d -> n (h d)", h=n_head, d=d_head),
            nn.Linear(n_head * d_head, d_node, bias=False),
        )

    def forward(self, node, pair, position_embedding, edge_index):
        normalized_node = self.node_norm(node)
        neighbor_node = normalized_node[edge_index]

        query = self.query_proj(normalized_node)
        key = self.key_proj(neighbor_node)
        value = self.value_proj(neighbor_node)
        query, key = qk_modulation(query, key, position_embedding, edge_index)

        attention_scores = (
            torch.einsum("n h d, n k h d -> n k h", query, key) / self.scale
            + self.attention_bias(pair)
        )
        attention_weights = torch.softmax(attention_scores, dim=1)
        gated_value = self.gate(normalized_node) * torch.einsum(
            "n k h, n k h d -> n h d",
            attention_weights,
            value,
        )
        return self.output_proj(gated_value)


class PairOuterProductMean(nn.Module):
    def __init__(self, d_node: int, d_edge: int, channel_dim: int = 32):
        super().__init__()
        self.channel_dim = channel_dim
        self.proj = nn.Linear(d_node, channel_dim * 2)
        self.out_proj = nn.Linear(channel_dim**2, d_edge, bias=False)
        self.pair_norm = nn.LayerNorm(d_edge)

    def forward(self, node, pair, edge_index):
        left, right = self.proj(node).chunk(2, dim=-1)
        right_neighbors = right[edge_index]
        pair_update = torch.einsum("n i, n k j -> n k i j", left, right_neighbors)
        pair_update = einops.rearrange(
            pair_update,
            "n k i j -> n k (i j)",
            i=self.channel_dim,
            j=self.channel_dim,
        )
        pair_update = self.out_proj(pair_update)
        return self.pair_norm(pair_update + math.sqrt(2) * pair)


class PairNeighborhoodAttention(nn.Module):
    def __init__(self, d_edge: int, n_head: int, d_head: int):
        super().__init__()
        self.scale = math.sqrt(d_head)
        self.pair_norm = nn.LayerNorm(d_edge)
        self.query_proj = nn.Sequential(
            nn.Linear(d_edge, n_head * d_head, bias=False),
            Rearrange("n k (h d) -> n k h d", h=n_head, d=d_head),
        )
        self.key_proj = nn.Sequential(
            nn.Linear(d_edge, n_head * d_head, bias=False),
            Rearrange("n k (h d) -> n k h d", h=n_head, d=d_head),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(d_edge, n_head * d_head, bias=False),
            Rearrange("n k (h d) -> n k h d", h=n_head, d=d_head),
        )
        self.attention_bias = nn.Linear(d_edge, n_head, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_edge, n_head * d_head, bias=True),
            Rearrange("n k (h d) -> n k h d", h=n_head, d=d_head),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Sequential(
            Rearrange("n k h d -> n k (h d)", h=n_head, d=d_head),
            nn.Linear(n_head * d_head, d_edge, bias=False),
        )

    def forward(self, pair, edge_index):
        normalized_pair = self.pair_norm(pair)
        self_index = torch.arange(len(pair), device=pair.device).unsqueeze(1).expand_as(edge_index)

        query = self.query_proj(normalized_pair)
        key = self.key_proj(normalized_pair)
        value = self.value_proj(normalized_pair)

        key = torch.cat([key[self_index], key[edge_index]], dim=2)
        value = torch.cat([value[self_index], value[edge_index]], dim=2)
        bias = self.attention_bias(normalized_pair)
        bias = torch.cat([bias[self_index], bias[edge_index]], dim=2)

        attention_scores = (
            torch.einsum("n k h d, n k j h d -> n k j h", query, key) / self.scale
            + bias
        )
        attention_weights = torch.softmax(attention_scores, dim=2)
        gated_value = self.gate(normalized_pair) * torch.einsum(
            "n k j h, n k j h d -> n k h d",
            attention_weights,
            value,
        )
        return self.output_proj(gated_value)


class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_head: int,
        n_head: int,
        use_sequence_attention: bool = False,
        d_seq: int = 1280,
        d_seq_na: int | None = None,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.use_sequence_attention = use_sequence_attention

        self.node_attention = NodeNeighborhoodAttention(
            d_node=d_node,
            d_edge=d_edge,
            n_head=n_head,
            d_head=d_head,
        )
        if self.use_sequence_attention:
            self.sequence_attention = SequenceAttention(
                d=d_node,
                d_seq=d_seq,
                d_seq_na=d_seq_na,
                d_head=d_head,
                n_head=n_head,
                checkpoint=checkpoint,
            )
        self.node_transition = GatedTransition(d_node)
        self.pair_outer_product = PairOuterProductMean(
            d_node=d_node,
            d_edge=d_edge,
            channel_dim=max(16, d_head // 2),
        )
        self.pair_attention = PairNeighborhoodAttention(
            d_edge=d_edge,
            n_head=n_head,
            d_head=max(8, (d_head // 3) * 2),
        )
        self.pair_transition = GatedTransition(d_edge)

    def forward(
        self,
        node,
        pair,
        position_embedding,
        edge_index,
        prot_mask=None,
        batch=None,
        attention_batch_size=200,
        prot_seq_emb=None,
        prot_seq_mask=None,
        na_seq_emb=None,
        na_seq_mask=None,
    ):
        node = node + self.node_attention(node, pair, position_embedding, edge_index)

        if self.use_sequence_attention:
            node = self.sequence_attention(
                node,
                prot_mask=prot_mask,
                batch=batch,
                attention_batch_size=attention_batch_size,
                prot_seq_emb=prot_seq_emb,
                prot_seq_mask=prot_seq_mask,
                na_seq_emb=na_seq_emb,
                na_seq_mask=na_seq_mask,
            )

        node = self.node_transition(node)
        pair = self.pair_outer_product(node, pair, edge_index)
        pair = pair + self.pair_attention(pair, edge_index)
        pair = self.pair_transition(pair)
        return node, pair
