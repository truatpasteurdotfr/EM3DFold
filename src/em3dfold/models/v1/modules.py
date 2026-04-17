import numpy as np
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels, freq_inv=100):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        # Project positions to sinusoidal embedding.
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq.to(tensor.device))
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x


def qk_modulation(q, k, pos_emb, edge_index):
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # N head_dim
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # N head_dim
    q_rot = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
    k_rot = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)
    q_new = q * cos_pos[:, None] + q_rot * sin_pos[:, None]
    k_new = k * cos_pos[edge_index][:, :, None] + k_rot * sin_pos[edge_index][:, :, None]
    return q_new, k_new


