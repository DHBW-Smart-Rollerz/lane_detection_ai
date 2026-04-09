import pdb

import torch
import torch.nn.modules
import torchvision

from model import layer


class vgg16bn(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(vgg16bn, self).__init__()
        model = list(
            torchvision.models.vgg16_bn(pretrained=pretrained).features.children()
        )
        model = model[:33] + model[34:43]
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained=False):
        super(resnet, self).__init__()
        self._is_mobilenet_v3 = False
        if layers == "9":
            block = torchvision.models.resnet.BasicBlock
            layers = [1, 1, 1, 1]
            model = torchvision.models.ResNet(block, layers)
        elif layers == "18":
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == "34":
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == "50":
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == "101":
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == "152":
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == "50next":
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == "101next":
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == "50wide":
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == "101wide":
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        elif layers == "34fca":
            model = torch.hub.load("cfzd/FcaNet", "fca34", pretrained=True)
        elif layers == "mobilenet-v3-small":
            self._is_mobilenet_v3 = True
            weights = (
                torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                if pretrained
                else None
            )
            model = torchvision.models.mobilenet_v3_small(
                weights=weights
            )
        elif layers == "mobilenet-v3-large":
            self._is_mobilenet_v3 = True
            weights = (
                torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
                if pretrained
                else None
            )
            model = torchvision.models.mobilenet_v3_large(
                weights=weights
            )
        else:
            raise NotImplementedError

        if self._is_mobilenet_v3:
            self.features = model.features
        else:
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4

    def forward(self, x):
        if self._is_mobilenet_v3:
            in_h, in_w = x.shape[-2], x.shape[-1]
            x2 = None
            x3 = None
            for block in self.features:
                x = block(x)
                stride_h = max(1, in_h // x.shape[-2])
                stride_w = max(1, in_w // x.shape[-1])
                if x2 is None and stride_h >= 8 and stride_w >= 8:
                    x2 = x
                if x3 is None and stride_h >= 16 and stride_w >= 16:
                    x3 = x
            x4 = x
            if x2 is None:
                x2 = x4
            if x3 is None:
                x3 = x4
            return x2, x3, x4

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4

        x = self.model.forward(x)
        # x = self.model.last_layer(x)
        return x, x, x.view(-1, 24, 6, 8)
