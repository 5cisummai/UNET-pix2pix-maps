from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator used in Pix2Pix."""

    def __init__(self, source_channels: int = 3, target_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        in_channels = source_channels + target_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([source_image, target_image], dim=1))
