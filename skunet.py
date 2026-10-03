"""
========
PyTorch implementation of SKUNet for semantic segmentation.
========

Reference
---------
Ramos et al., "Leveraging U-Net and selective feature extraction for land cover classification using remote sensing imagery", Scientific Reports, 2025.
"""


from __future__ import annotations

from typing import List, Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UnetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode="nearest",
        )
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        return feature_map


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
    ) -> None:
        super().__init__()

        encoder_channels = list(encoder_channels[1:])[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]

        self.blocks = nn.ModuleList(
            [
                UnetDecoderBlock(block_in, block_skip, block_out)
                for block_in, block_skip, block_out in zip(
                    in_channels,
                    skip_channels,
                    decoder_channels,
                )
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        spatial_shapes = [feature.shape[2:] for feature in features][::-1]
        features = features[1:][::-1]

        x = features[0]
        skip_connections = features[1:]

        for index, decoder_block in enumerate(self.blocks):
            height, width = spatial_shapes[index + 1]
            skip = (
                skip_connections[index]
                if index < len(skip_connections)
                else None
            )
            x = decoder_block(x, height, width, skip_connection=skip)

        return x


class SKUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1.")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1.")

        self.encoder = timm.create_model(
            "skresnext50_32x4d",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=3,
        )
        if in_channels != 3:
            self._adapt_input_convolution(in_channels)

        encoder_channels = [in_channels, 64, 256, 512, 1024, 2048]
        decoder_channels = (256, 128, 64, 32, 16)
        self.decoder = UnetDecoder(encoder_channels, decoder_channels)
        self.segmentation_head = nn.Conv2d(
            decoder_channels[-1],
            num_classes,
            kernel_size=3,
            padding=1,
        )

        self._initialize_decoder()
        self._initialize_head()

    def _adapt_input_convolution(self, in_channels: int) -> None:
        old_conv = self.encoder.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        )
        with torch.no_grad():
            mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_rgb.repeat(1, in_channels, 1, 1))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.encoder.conv1 = new_conv

    def _initialize_decoder(self) -> None:
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _initialize_head(self) -> None:
        nn.init.xavier_uniform_(self.segmentation_head.weight)
        if self.segmentation_head.bias is not None:
            nn.init.constant_(self.segmentation_head.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x] + list(self.encoder(x))
        x = self.decoder(features)
        return self.segmentation_head(x)


__all__ = ["SKUNet"]
