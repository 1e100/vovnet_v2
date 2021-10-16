""" VoVNet as per https://arxiv.org/pdf/1904.09730.pdf (v1) and
https://arxiv.org/pdf/1911.06667.pdf (v2). This is adaptation to
1D inputs (signals)."""

from typing import Union, List
import collections

import torch
from torch import nn

# The paper is unclear as to where to downsample, so the downsampling was
# derived from the pretrained model graph as visualized by Netron. V2 simply
# enables ESE and identity connections here, nothing else changes.
#
# Note regarding 1D: these configurations were just transplanted straight
# across to 1D case, so they are probably not ideal for the specific 1D
# use case.
CONFIG = {
    # Introduced in V2. Difference is 3 repeats instead of 5 within each block.
    "vovnet19_1d": [
        # kernel size, inner channels, layer repeats, output channels, downsample
        [3, 64, 3, 128, True],
        [3, 80, 3, 256, True],
        [3, 96, 3, 348, True],
        [3, 112, 3, 512, True],
    ],
    "vovnet27_slim_1d": [
        [3, 64, 5, 128, True],
        [3, 80, 5, 256, True],
        [3, 96, 5, 348, True],
        [3, 112, 5, 512, True],
    ],
    "vovnet39_1d": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],
        [3, 192, 5, 768, True],  # x2
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x2
        [3, 224, 5, 1024, False],
    ],
    "vovnet57_1d": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],
        [3, 192, 5, 768, True],  # x4
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x3
        [3, 224, 5, 1024, False],
        [3, 224, 5, 1024, False],
    ],
}


class _ESE(nn.Module):
    def __init__(self, channels: int) -> None:
        # TODO: Might want to experiment with bias=False.
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean([2], keepdim=True)
        y = self.conv(y)
        # Hard sigmoid multiplied by input.
        return x * (nn.functional.relu6(y + 3, inplace=True) / 6.0)


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class _DSConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv1d(
                in_ch,
                in_ch,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class _OSA(nn.Module):
    def __init__(
        self,
        in_ch: int,
        inner_ch: int,
        out_ch: int,
        repeats: int = 5,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: bool = False,
        use_dsconv: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        osa_conv = _DSConvBnRelu if use_dsconv else _ConvBnRelu
        self.layers = nn.ModuleList(
            [
                osa_conv(
                    in_ch if r == 0 else inner_ch,
                    inner_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for r in range(repeats)
            ]
        )
        self.exit_conv = _ConvBnRelu(in_ch + repeats * inner_ch, out_ch, kernel_size=1)
        self.ese = _ESE(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through all modules, but retain outputs.
        input = x
        if self.downsample:
            x = nn.functional.max_pool1d(x, 3, stride=2, padding=1)
        features = [x]
        for layer in self.layers:
            features.append(layer(x))
            x = features[-1]
        x = torch.cat(features, dim=1)
        x = self.exit_conv(x)
        x = self.ese(x)
        # All non-downsampling V2 layers have a residual. They also happen to
        # not change the number of channels.
        if not self.downsample:
            x += input
        return x


class VoVNet1D(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        model_type: str = "vovnet19_1d",
        has_classifier: bool = True,
        dropout: float = 0.2,
        use_dsconv: bool = False,
    ):
        """Usage:
        >>> net = VoVNet1D(1, 2, use_dsconv=True)
        >>> net = net.eval()
        >>> with torch.no_grad():
        ...     y = net(torch.rand(2, 1, 64))
        >>> print(list(y.shape))
        [2, 2]
        """
        super().__init__()

        stem_conv = _DSConvBnRelu if use_dsconv else _ConvBnRelu

        # Input stage.
        self.stem = nn.Sequential(
            _ConvBnRelu(input_channels, 64, kernel_size=3, stride=2),
            stem_conv(64, 64, kernel_size=3, stride=1),
            stem_conv(64, 128, kernel_size=3, stride=1),
        )

        body_layers = collections.OrderedDict()
        conf = CONFIG[model_type]
        in_ch = 128
        feature_channels = []
        for idx, block in enumerate(conf):
            kernel_size, inner_ch, repeats, out_ch, downsample = block
            body_layers[f"osa{idx}"] = _OSA(
                in_ch,
                inner_ch,
                out_ch,
                repeats=repeats,
                kernel_size=kernel_size,
                downsample=downsample,
                use_dsconv=use_dsconv,
            )
            if downsample:
                feature_channels.append(in_ch)
            in_ch = out_ch
        # Last layer output depth.
        feature_channels.append(out_ch)
        self.body = nn.Sequential(body_layers)
        self.feature_channels = feature_channels

        if has_classifier:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_ch, num_classes, bias=True),
            )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        y = self.stem(x)
        # Return features before downsampling for e.g. detection and segmentation.
        if return_features:
            features = []
            for block in self.body:
                if block.downsample:
                    features.append(y)
                y = block(y)
            features.append(y)
            return features
        else:
            y = self.body(y)
            if hasattr(self, "classifier"):
                y = self.classifier(y)
            return y
