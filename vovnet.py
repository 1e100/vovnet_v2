""" VoVNet as per https://arxiv.org/pdf/1904.09730.pdf. """

import collections

import torch
from torch import nn

# The paper is unclear as to where to downsample, so the downsampling was
# derived from the pretrained model graph as visualized by Netron.
CONFIG = {
    "vovnet27_slim": [
        # kernel size, inner channels, layer repeats, output channels, downsample
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
}


class ConvBnRelu(nn.Sequential):
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
                ConvBnRelu(
                    in_ch if r == 0 else inner_ch,
                    inner_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for r in range(repeats)
            ]
        )
        self.exit_conv = ConvBnRelu(in_ch + repeats * inner_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through all modules, but retain outputs.
        if self.downsample:
            x = nn.functional.max_pool2d(x, 3, stride=2, padding=1)
        features = [x]
        for l in self.layers:
            features.append(l(x))
            x = features[-1]
        return self.exit_conv(torch.cat(features, dim=1))


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
            ConvBnRelu(in_ch, 64, kernel_size=3, stride=2),
            ConvBnRelu(64, 64, kernel_size=3, stride=1),
            ConvBnRelu(64, 128, kernel_size=3, stride=1),
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
