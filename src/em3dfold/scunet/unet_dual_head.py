from __future__ import annotations

import torch
import torch.nn as nn

from em3dfold.scunet.frn import FilterResponseNorm3d


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            FilterResponseNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            FilterResponseNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.block = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


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


class UNetDualHead(nn.Module):
    def __init__(
        self,
        in_nc: int = 1,
        dim: int = 32,
        seg_classes: int = 3,
        atom_channels: int = 4,
        **_: object,
    ) -> None:
        super().__init__()
        self.seg_classes = int(seg_classes)
        self.atom_channels = int(atom_channels)
        base = int(dim)

        self.stem = _ConvBlock(in_nc, base)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = _ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = _ConvBlock(base * 4, base * 8)

        self.up3 = _UpBlock(base * 8, base * 4, base * 4)
        self.up2 = _UpBlock(base * 4, base * 2, base * 2)
        self.up1 = _UpBlock(base * 2, base, base)

        self.seg_decoder = _DecoderHead(base, base)
        self.seg_head = nn.Conv3d(base, self.seg_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.atom_decoder = _DecoderHead((2 * base) + self.seg_classes, base)
        self.atom_head = nn.Conv3d(base, self.atom_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x0: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.stem(x0)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        xb = self.bottleneck(self.pool3(x3))

        x = self.up3(xb, x3)
        x = self.up2(x, x2)
        trunk = self.up1(x, x1)

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
