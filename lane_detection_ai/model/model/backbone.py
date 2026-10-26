import torch
import torchvision

_RESNETS = {
    "18": torchvision.models.resnet18,
    "34": torchvision.models.resnet34,
    "50": torchvision.models.resnet50,
    "101": torchvision.models.resnet101,
    "152": torchvision.models.resnet152,
    "50next": torchvision.models.resnext50_32x4d,
    "101next": torchvision.models.resnext101_32x8d,
    "50wide": torchvision.models.wide_resnet50_2,
    "101wide": torchvision.models.wide_resnet101_2,
}


class ResNetBackbone(torch.nn.Module):
    """ResNet feature extractor returning the outputs of layer 2, 3 and 4."""

    def __init__(self, layers: str, *, pretrained: bool = False) -> None:
        """Initialize the backbone.

        Args:
            layers: The ResNet variant, e.g. "18", "34" or "50next".
            pretrained: Load the ImageNet weights of torchvision (downloads them on
                first use). Not required when the weights of a trained lane
                detection checkpoint are loaded afterwards.
        """
        super().__init__()
        if layers == "9":
            model = torchvision.models.ResNet(
                torchvision.models.resnet.BasicBlock, [1, 1, 1, 1]
            )
        elif layers in _RESNETS:
            model = _RESNETS[layers](weights="DEFAULT" if pretrained else None)
        else:
            msg = f"Unsupported backbone: {layers!r}"
            raise NotImplementedError(msg)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the features of layer 2, 3 and 4."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4
