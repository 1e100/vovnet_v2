""" VoVNet as per https://arxiv.org/pdf/1904.09730.pdf (v1) and
https://arxiv.org/pdf/1911.06667.pdf (v2). """

import collections

import torch
from torch import nn

# The paper is unclear as to where to downsample, so the downsampling was
# derived from the pretrained model graph as visualized by Netron. V2 simply
# enables ESE and identity connections here, nothing else changes.
CONFIG = {
    # Introduced in V2. Difference is 3 repeats instead of 5 within each block.
    "vovnet19": [
        # kernel size, inner channels, layer repeats, output channels, downsample
        [3, 64, 3, 128, True],
        [3, 80, 3, 256, True],
        [3, 96, 3, 348, True],
        [3, 112, 3, 512, True],
    ],
    "vovnet27_slim": [
        [3, 64, 5, 128, True],
        [3, 80, 5, 256, True],
        [3, 96, 5, 348, True],
        [3, 112, 5, 512, True],
    ],
    "vovnet39": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],
        [3, 192, 5, 768, True],  # x2
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x2
        [3, 224, 5, 1024, False],
    ],
    "vovnet57": [
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
    "vovnet99": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],  # x3
        [3, 160, 5, 512, False],
        [3, 160, 5, 512, False],
        [3, 192, 5, 768, True],  # x9
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
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
        # TODO: Might want to experiment with bias=False. At least for
        # MobileNetV3 it leads to better accuracy on detection.
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean([2, 3], keepdim=True)
        y = self.conv(y)
        # Hard sigmoid multiplied by input.
        return x * (nn.functional.relu6(y + 3, inplace=True) / 6.0)


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
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
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.layers = nn.ModuleList(
            [
                _ConvBnRelu(
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
            x = nn.functional.max_pool2d(x, 3, stride=2, padding=1)
        features = [x]
        for l in self.layers:
            features.append(l(x))
            x = features[-1]
        x = torch.cat(features, dim=1)
        x = self.exit_conv(x)
        x = self.ese(x)
        # All non-downsampling V2 layers have a residual. They also happen to
        # not change the number of channels.
        if not self.downsample:
            x += input
        return x


class VoVNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        num_classes: int = 1000,
        model_type: str = "vovnet39",
        has_classifier: bool = True,
        dropout: float = 0.2,
    ):
        """ Usage:
        >>> net = VoVNet(3, 1000)
        >>> net = net.eval()
        >>> with torch.no_grad():
        ...     y = net(torch.rand(2, 3, 64, 64))
        >>> print(list(y.shape))
        [2, 1000]
        """
        super().__init__()

        # Input stage.
        self.stem = nn.Sequential(
            _ConvBnRelu(in_ch, 64, kernel_size=3, stride=2),
            _ConvBnRelu(64, 64, kernel_size=3, stride=1),
            _ConvBnRelu(64, 128, kernel_size=3, stride=1),
        )

        body_layers = collections.OrderedDict()
        conf = CONFIG[model_type]
        in_ch = 128
        for idx, block in enumerate(conf):
            kernel_size, inner_ch, repeats, out_ch, downsample = block
            body_layers[f"osa{idx}"] = _OSA(
                in_ch,
                inner_ch,
                out_ch,
                repeats=repeats,
                kernel_size=kernel_size,
                downsample=downsample,
            )
            in_ch = out_ch
        self.body = nn.Sequential(body_layers)
        self.has_classifier = has_classifier

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_ch, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.body(y)
        if self.has_classifier:
            y = self.classifier(y)
        return y
