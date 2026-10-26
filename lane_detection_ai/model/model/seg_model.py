import torch

from lane_detection_ai.model.utils.common import initialize_weights


class ConvBnRelu(torch.nn.Module):
    """Convolution followed by batch norm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        """Initialize the block (same arguments as `torch.nn.Conv2d`)."""
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch norm and ReLU."""
        return self.relu(self.bn(self.conv(x)))


class SegHead(torch.nn.Module):
    """Auxiliary segmentation head (only used for training)."""

    def __init__(self, backbone: str, num_lanes: int) -> None:
        """Initialize the segmentation head.

        Args:
            backbone: The ResNet variant of the backbone.
            num_lanes: Number of lanes to segment.
        """
        super().__init__()
        small_backbone = backbone in {"34", "18"}

        self.aux_header2 = torch.nn.Sequential(
            ConvBnRelu(128 if small_backbone else 512, 128, 3, padding=1),
            ConvBnRelu(128, 128, 3, padding=1),
            ConvBnRelu(128, 128, 3, padding=1),
            ConvBnRelu(128, 128, 3, padding=1),
        )
        self.aux_header3 = torch.nn.Sequential(
            ConvBnRelu(256 if small_backbone else 1024, 128, 3, padding=1),
            ConvBnRelu(128, 128, 3, padding=1),
            ConvBnRelu(128, 128, 3, padding=1),
        )
        self.aux_header4 = torch.nn.Sequential(
            ConvBnRelu(512 if small_backbone else 2048, 128, 3, padding=1),
            ConvBnRelu(128, 128, 3, padding=1),
        )
        self.aux_combine = torch.nn.Sequential(
            ConvBnRelu(384, 256, 3, padding=2, dilation=2),
            ConvBnRelu(256, 128, 3, padding=2, dilation=2),
            ConvBnRelu(128, 128, 3, padding=2, dilation=2),
            ConvBnRelu(128, 128, 3, padding=4, dilation=4),
            # Output: n, num_lanes + 1, h, w
            torch.nn.Conv2d(128, num_lanes + 1, 1),
        )

        initialize_weights(
            self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine
        )

    def forward(
        self, x2: torch.Tensor, x3: torch.Tensor, fea: torch.Tensor
    ) -> torch.Tensor:
        """Return the segmentation logits."""
        x2 = self.aux_header2(x2)
        x3 = self.aux_header3(x3)
        x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode="bilinear")
        x4 = self.aux_header4(fea)
        x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode="bilinear")
        aux_seg = torch.cat([x2, x3, x4], dim=1)
        return self.aux_combine(aux_seg)
