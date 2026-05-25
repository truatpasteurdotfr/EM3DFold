import math
import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor

from em3dfold.scunet.frn import FilterResponseNorm3d


try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func = None


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


torch.set_printoptions(threshold=np.inf)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class WMSA(nn.Module):
    """Window attention backed by flash-attn.

    The surrounding SCUNet structure is unchanged. We preserve:
    - window partitioning
    - shifted-window connectivity
    - learnable relative position bias

    The only changed part is the attention kernel itself.
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1) * (2 * window_size - 1),
                self.n_heads,
            )
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(
                2 * window_size - 1,
                2 * window_size - 1,
                2 * window_size - 1,
                self.n_heads,
            )
            .transpose(2, 3)
            .transpose(1, 2)
            .transpose(0, 1)
            .reshape(
                self.n_heads,
                2 * window_size - 1,
                2 * window_size - 1,
                2 * window_size - 1,
            )
        )

        cord = torch.tensor(
            np.array(
                [
                    [i, j, k]
                    for i in range(self.window_size)
                    for j in range(self.window_size)
                    for k in range(self.window_size)
                ]
            )
        )
        self.relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        self.tokens_per_window = self.window_size ** 3
        # Extra dims encode the relative bias exactly as additive qk logits.
        self.flash_head_dim = _round_up_to_multiple(
            self.head_dim + self.tokens_per_window,
            8,
        )

    def generate_mask(self, h, w, d, p, shift):
        attn_mask = torch.zeros(
            h,
            w,
            d,
            p,
            p,
            p,
            p,
            p,
            p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :, :s, :, :, s:, :, :] = True
        attn_mask[-1, :, :, s:, :, :, :s, :, :] = True
        attn_mask[:, -1, :, :, :s, :, :, s:, :] = True
        attn_mask[:, -1, :, :, s:, :, :, :s, :] = True
        attn_mask[:, :, -1, :, :, :s, :, :, s:] = True
        attn_mask[:, :, -1, :, :, s:, :, :, :s] = True
        attn_mask = rearrange(
            attn_mask,
            "w1 w2 w3 p1 p2 p3 p4 p5 p6 -> 1 1 (w1 w2 w3) (p1 p2 p3) (p4 p5 p6)",
        )
        return attn_mask

    @staticmethod
    @lru_cache(maxsize=64)
    def _cached_window_groups(attn_type: str, h: int, w: int, d: int, p: int):
        tokens_per_window = p ** 3
        if attn_type == "W":
            group = tuple(range(tokens_per_window))
            return tuple((group,) for _ in range(h * w * d))

        # Rebuild the same mask pattern as the reference path, but cache the
        # resulting connected groups as Python tuples to avoid per-forward work.
        attn_mask = torch.zeros(h, w, d, p, p, p, p, p, p, dtype=torch.bool)
        shift = p // 2
        s = p - shift
        attn_mask[-1, :, :, :s, :, :, s:, :, :] = True
        attn_mask[-1, :, :, s:, :, :, :s, :, :] = True
        attn_mask[:, -1, :, :, :s, :, :, s:, :] = True
        attn_mask[:, -1, :, :, s:, :, :, :s, :] = True
        attn_mask[:, :, -1, :, :, :s, :, :, s:] = True
        attn_mask[:, :, -1, :, :, s:, :, :, :s] = True
        attn_mask = rearrange(
            attn_mask,
            "w1 w2 w3 p1 p2 p3 p4 p5 p6 -> (w1 w2 w3) (p1 p2 p3) (p4 p5 p6)",
        )

        groups_per_window = []
        for window_mask in attn_mask:
            allowed = ~window_mask
            remaining = set(range(tokens_per_window))
            window_groups = []
            while remaining:
                root = min(remaining)
                group = tuple(torch.nonzero(allowed[root], as_tuple=False).flatten().tolist())
                for idx in group:
                    remaining.discard(idx)
                window_groups.append(group)
            groups_per_window.append(tuple(window_groups))
        return tuple(groups_per_window)

    def _relative_bias(self, device, dtype):
        relation = self.relation.to(device=device)
        bias = self.relative_position_params[
            :,
            relation[:, :, 0].long(),
            relation[:, :, 1].long(),
            relation[:, :, 2].long(),
        ]
        return bias.to(dtype=dtype)

    def _forward_reference(self, x):
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(
                    -(self.window_size // 2),
                    -(self.window_size // 2),
                    -(self.window_size // 2),
                ),
                dims=(1, 2, 3),
            )
        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) (w3 p3) c -> b w1 w2 w3 p1 p2 p3 c",
            p1=self.window_size,
            p2=self.window_size,
            p3=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)
        d_windows = x.size(3)
        assert h_windows == w_windows == d_windows

        x = rearrange(
            x,
            "b w1 w2 w3 p1 p2 p3 c -> b (w1 w2 w3) (p1 p2 p3) c",
            p1=self.window_size,
            p2=self.window_size,
            p3=self.window_size,
        )

        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv,
            "b nw np (threeh c) -> threeh b nw np c",
            c=self.head_dim,
        ).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        sim = sim + rearrange(
            self._relative_bias(x.device, x.dtype),
            "h p q -> h 1 1 p q",
        )
        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows,
                w_windows,
                d_windows,
                self.window_size,
                shift=self.window_size // 2,
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2 w3) (p1 p2 p3) c -> b (w1 p1) (w2 p2) (w3 p3) c",
            w1=h_windows,
            w2=w_windows,
            w3=d_windows,
            p1=self.window_size,
            p2=self.window_size,
            p3=self.window_size,
        )
        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(
                    self.window_size // 2,
                    self.window_size // 2,
                    self.window_size // 2,
                ),
                dims=(1, 2, 3),
            )
        return output

    def forward(self, x):
        use_flash = (
            flash_attn_varlen_qkvpacked_func is not None
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
        )
        if not use_flash:
            return self._forward_reference(x)

        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(
                    -(self.window_size // 2),
                    -(self.window_size // 2),
                    -(self.window_size // 2),
                ),
                dims=(1, 2, 3),
            )

        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) (w3 p3) c -> b w1 w2 w3 p1 p2 p3 c",
            p1=self.window_size,
            p2=self.window_size,
            p3=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)
        d_windows = x.size(3)
        assert h_windows == w_windows == d_windows

        x = rearrange(
            x,
            "b w1 w2 w3 p1 p2 p3 c -> b (w1 w2 w3) (p1 p2 p3) c",
            p1=self.window_size,
            p2=self.window_size,
            p3=self.window_size,
        )
        bsz, n_windows, _, _ = x.shape

        qkv = self.embedding_layer(x).view(
            bsz,
            n_windows,
            self.tokens_per_window,
            3,
            self.n_heads,
            self.head_dim,
        )
        rel_bias = self._relative_bias(x.device, x.dtype)
        groups_per_window = self._cached_window_groups(
            self.type,
            h_windows,
            w_windows,
            d_windows,
            self.window_size,
        )

        packed_qkv = []
        meta = []
        seq_lens = [0]
        for batch_idx in range(bsz):
            for window_idx, groups in enumerate(groups_per_window):
                for group in groups:
                    idxs = torch.tensor(group, device=x.device, dtype=torch.long)
                    group_len = int(idxs.numel())
                    q_sel = qkv[batch_idx, window_idx, idxs, 0]
                    k_sel = qkv[batch_idx, window_idx, idxs, 1]
                    v_sel = qkv[batch_idx, window_idx, idxs, 2]

                    bias_group = rel_bias.index_select(1, idxs).index_select(2, idxs)
                    q_bias = q_sel.new_zeros(
                        group_len,
                        self.n_heads,
                        self.tokens_per_window,
                    )
                    q_bias.index_copy_(2, idxs, bias_group.permute(1, 0, 2))

                    basis = q_sel.new_zeros(group_len, self.tokens_per_window)
                    basis[torch.arange(group_len, device=x.device), idxs] = 1.0
                    k_bias = basis[:, None, :].expand(-1, self.n_heads, -1)

                    q_aug = q_sel.new_zeros(group_len, self.n_heads, self.flash_head_dim)
                    k_aug = k_sel.new_zeros(group_len, self.n_heads, self.flash_head_dim)
                    v_aug = v_sel.new_zeros(group_len, self.n_heads, self.flash_head_dim)

                    q_aug[..., : self.head_dim] = q_sel * self.scale
                    q_aug[..., self.head_dim : self.head_dim + self.tokens_per_window] = q_bias
                    k_aug[..., : self.head_dim] = k_sel
                    k_aug[..., self.head_dim : self.head_dim + self.tokens_per_window] = k_bias
                    v_aug[..., : self.head_dim] = v_sel

                    packed_qkv.append(torch.stack((q_aug, k_aug, v_aug), dim=1))
                    meta.append((batch_idx, window_idx, idxs))
                    seq_lens.append(seq_lens[-1] + group_len)

        packed_qkv = torch.cat(packed_qkv, dim=0).contiguous()
        cu_seqlens = torch.tensor(seq_lens, device=x.device, dtype=torch.int32)
        packed_out = flash_attn_varlen_qkvpacked_func(
            packed_qkv,
            cu_seqlens,
            max_seqlen=self.tokens_per_window,
            dropout_p=0.0,
            softmax_scale=1.0,
            causal=False,
        )

        output = x.new_zeros(
            bsz,
            n_windows,
            self.tokens_per_window,
            self.n_heads,
            self.head_dim,
        )
        offset = 0
        for batch_idx, window_idx, idxs in meta:
            group_len = int(idxs.numel())
            output[batch_idx, window_idx, idxs] = packed_out[
                offset : offset + group_len,
                :,
                : self.head_dim,
            ]
            offset += group_len

        output = output.reshape(bsz, n_windows, self.tokens_per_window, self.input_dim)
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2 w3) (p1 p2 p3) c -> b (w1 p1) (w2 p2) (w3 p3) c",
            w1=h_windows,
            w2=w_windows,
            w3=d_windows,
            p1=self.window_size,
            p2=self.window_size,
            p3=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(
                    self.window_size // 2,
                    self.window_size // 2,
                    self.window_size // 2,
                ),
                dims=(1, 2, 3),
            )
        return output


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type="W", input_resolution=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"]
        self.type = type
        if input_resolution <= window_size:
            self.type = "W"

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type="W", input_resolution=None):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ["W", "SW"]
        if self.input_resolution <= self.window_size:
            self.type = "W"

        self.trans_block = Block(
            self.trans_dim,
            self.trans_dim,
            self.head_dim,
            self.window_size,
            self.drop_path,
            self.type,
            self.input_resolution,
        )
        self.conv1_1 = nn.Conv3d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            1,
            1,
            0,
            bias=True,
        )
        self.conv1_2 = nn.Conv3d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            1,
            1,
            0,
            bias=True,
        )
        self.conv_block = nn.Sequential(
            nn.Conv3d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            FilterResponseNorm3d(self.conv_dim),
            nn.Conv3d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            FilterResponseNorm3d(self.conv_dim),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange("b c h w d -> b h w d c")(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange("b h w d c -> b c h w d")(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


class SCUNet(nn.Module):
    def __init__(
        self,
        in_nc=1,
        config=[2, 2, 2, 2, 2, 2, 2],
        dim=32,
        drop_path_rate=0.2,
        input_resolution=48,
        head_dim=16,
        window_size=3,
        n_classes=1,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        self.m_head = [nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [
            ConvTransBlock(
                dim // 2,
                dim // 2,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution,
            )
            for i in range(config[0])
        ] + [nn.Conv3d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [
            ConvTransBlock(
                dim,
                dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 2,
            )
            for i in range(config[1])
        ] + [nn.Conv3d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [
            ConvTransBlock(
                2 * dim,
                2 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 4,
            )
            for i in range(config[2])
        ] + [nn.Conv3d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [
            ConvTransBlock(
                4 * dim,
                4 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 8,
            )
            for i in range(config[3])
        ]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose3d(8 * dim, 4 * dim, 2, 2, 0, bias=False)] + [
            ConvTransBlock(
                2 * dim,
                2 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 4,
            )
            for i in range(config[4])
        ]

        begin += config[4]
        self.m_up2 = [nn.ConvTranspose3d(4 * dim, 2 * dim, 2, 2, 0, bias=False)] + [
            ConvTransBlock(
                dim,
                dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 2,
            )
            for i in range(config[5])
        ]

        begin += config[5]
        self.m_up1 = [nn.ConvTranspose3d(2 * dim, dim, 2, 2, 0, bias=False)] + [
            ConvTransBlock(
                dim // 2,
                dim // 2,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution,
            )
            for i in range(config[6])
        ]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)

        self.m_tail = nn.Sequential(nn.Conv3d(dim, n_classes, 3, 1, 1, bias=False))

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        y = self.m_tail(x + x1)
        return y

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
