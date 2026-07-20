from __future__ import annotations

import torch
import torch.nn as nn

from em3dfold.scunet.frn import FilterResponseNorm3d
from em3dfold.scunet.scunet import ConvTransBlock


class _DecoderHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            FilterResponseNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            FilterResponseNorm3d(hidden_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SCUNetDualHead(nn.Module):
    def __init__(
        self,
        in_nc: int = 1,
        config: list[int] | tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2),
        dim: int = 32,
        drop_path_rate: float = 0.2,
        input_resolution: int = 48,
        head_dim: int = 16,
        window_size: int = 3,
        seg_classes: int = 3,
        atom_channels: int = 4,
        **_: object,
    ) -> None:
        super().__init__()
        self.config = list(config)
        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.window_size = int(window_size)
        self.seg_classes = int(seg_classes)
        self.atom_channels = int(atom_channels)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.config))]

        self.m_head = nn.Sequential(nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False))

        begin = 0
        self.m_down1 = nn.Sequential(
            *[
                ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin], "W" if not i % 2 else "SW", input_resolution)
                for i in range(self.config[0])
            ],
            nn.Conv3d(dim, 2 * dim, 2, 2, 0, bias=False),
        )

        begin += self.config[0]
        self.m_down2 = nn.Sequential(
            *[
                ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 2)
                for i in range(self.config[1])
            ],
            nn.Conv3d(2 * dim, 4 * dim, 2, 2, 0, bias=False),
        )

        begin += self.config[1]
        self.m_down3 = nn.Sequential(
            *[
                ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 4)
                for i in range(self.config[2])
            ],
            nn.Conv3d(4 * dim, 8 * dim, 2, 2, 0, bias=False),
        )

        begin += self.config[2]
        self.m_body = nn.Sequential(
            *[
                ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 8)
                for i in range(self.config[3])
            ]
        )

        begin += self.config[3]
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose3d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
            *[
                ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 4)
                for i in range(self.config[4])
            ],
        )

        begin += self.config[4]
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose3d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
            *[
                ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin], "W" if not i % 2 else "SW", input_resolution // 2)
                for i in range(self.config[5])
            ],
        )

        begin += self.config[5]
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose3d(2 * dim, dim, 2, 2, 0, bias=False),
            *[
                ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin], "W" if not i % 2 else "SW", input_resolution)
                for i in range(self.config[6])
            ],
        )

        self.seg_decoder = _DecoderHead(dim, dim)
        self.seg_head = nn.Conv3d(dim, self.seg_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.atom_decoder = _DecoderHead((2 * dim) + self.seg_classes, dim)
        self.atom_head = nn.Conv3d(dim, self.atom_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x0: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)

        trunk = x + x1
        seg_feat = self.seg_decoder(trunk)
        seg_logits = self.seg_head(seg_feat)
        seg_feat_for_atom = seg_feat.detach()
        with torch.no_grad():
            seg_probs = torch.softmax(seg_logits, dim=1)

        atom_feat = self.atom_decoder(torch.cat([trunk, seg_feat_for_atom, seg_probs], dim=1))
        atom_logits = self.atom_head(atom_feat)

        return {
            "seg_logits": seg_logits,
            "atom_logits": atom_logits,
        }
