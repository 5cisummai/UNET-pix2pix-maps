from __future__ import annotations

import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_norm),
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNetGenerator(nn.Module):
    """Pix2Pix-style U-Net generator for paired image translation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.encoder1 = DownBlock(in_channels, base_channels, use_norm=False)
        self.encoder2 = DownBlock(base_channels, base_channels * 2)
        self.encoder3 = DownBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = DownBlock(base_channels * 4, base_channels * 8)
        self.encoder5 = DownBlock(base_channels * 8, base_channels * 8)
        self.encoder6 = DownBlock(base_channels * 8, base_channels * 8)
        self.encoder7 = DownBlock(base_channels * 8, base_channels * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = UpBlock(base_channels * 8, base_channels * 8, use_dropout=True)
        self.decoder2 = UpBlock(base_channels * 16, base_channels * 8, use_dropout=True)
        self.decoder3 = UpBlock(base_channels * 16, base_channels * 8, use_dropout=True)
        self.decoder4 = UpBlock(base_channels * 16, base_channels * 8)
        self.decoder5 = UpBlock(base_channels * 16, base_channels * 4)
        self.decoder6 = UpBlock(base_channels * 8, base_channels * 2)
        self.decoder7 = UpBlock(base_channels * 4, base_channels)

        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)
        encoder6 = self.encoder6(encoder5)
        encoder7 = self.encoder7(encoder6)

        bottleneck = self.bottleneck(encoder7)

        decoder1 = self.decoder1(bottleneck)
        decoder2 = self.decoder2(torch.cat([decoder1, encoder7], dim=1))
        decoder3 = self.decoder3(torch.cat([decoder2, encoder6], dim=1))
        decoder4 = self.decoder4(torch.cat([decoder3, encoder5], dim=1))
        decoder5 = self.decoder5(torch.cat([decoder4, encoder4], dim=1))
        decoder6 = self.decoder6(torch.cat([decoder5, encoder3], dim=1))
        decoder7 = self.decoder7(torch.cat([decoder6, encoder2], dim=1))

        return self.output_block(torch.cat([decoder7, encoder1], dim=1))
