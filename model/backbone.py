import pdb
import os

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
        self._is_yolov5 = False
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
        elif layers in ["yolov5n", "yolov5s"]:
            self._is_yolov5 = True
            # Allow using a local clone via YOLOV5_REPO=/path/to/yolov5 for offline clusters.
            repo = os.environ.get("YOLOV5_REPO", "ultralytics/yolov5")
            source = "local" if os.path.isdir(repo) else "github"
            try:
                model = torch.hub.load(
                    repo,
                    layers,
                    pretrained=pretrained,
                    autoshape=False,
                    source=source,
                    trust_repo=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load YOLOv5 backbone '{layers}' from '{repo}' (source={source}). "
                    "Either set YOLOV5_REPO to a local YOLOv5 clone or ensure network access for torch.hub."
                ) from exc
        else:
            raise NotImplementedError

        if self._is_mobilenet_v3:
            self.features = model.features
        elif self._is_yolov5:
            self.yolo_model = model
            self.yolo_feature_tensors = []
            self._yolo_hook_handles = []

            if hasattr(self.yolo_model, "model") and hasattr(self.yolo_model.model, "model"):
                hook_modules = list(self.yolo_model.model.model)
            elif hasattr(self.yolo_model, "model") and isinstance(
                self.yolo_model.model, (torch.nn.Sequential, torch.nn.ModuleList)
            ):
                hook_modules = list(self.yolo_model.model)
            else:
                hook_modules = [self.yolo_model]

            for module in hook_modules:
                self._yolo_hook_handles.append(
                    module.register_forward_hook(self._capture_yolo_feature)
                )
        else:
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4

    def _capture_yolo_feature(self, module, inputs, output):
        stack = [output]
        while stack:
            item = stack.pop()
            if torch.is_tensor(item):
                if item.dim() == 4:
                    self.yolo_feature_tensors.append(item)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
            elif isinstance(item, dict):
                stack.extend(item.values())

    def _pick_stride_features(self, feature_tensors, in_h, in_w):
        x2 = None
        x3 = None
        x4 = None
        best_x4_stride = -1
        for fea in feature_tensors:
            if fea.shape[-2] <= 0 or fea.shape[-1] <= 0:
                continue
            stride_h = max(1, in_h // fea.shape[-2])
            stride_w = max(1, in_w // fea.shape[-1])
            stride = min(stride_h, stride_w)
            if x2 is None and stride >= 8:
                x2 = fea
            if x3 is None and stride >= 16:
                x3 = fea
            if stride >= 32 and stride > best_x4_stride:
                best_x4_stride = stride
                x4 = fea

        if x4 is None and feature_tensors:
            x4 = feature_tensors[-1]
        if x2 is None:
            x2 = x4
        if x3 is None:
            x3 = x4
        return x2, x3, x4

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

        if self._is_yolov5:
            in_h, in_w = x.shape[-2], x.shape[-1]
            self.yolo_feature_tensors = []
            _ = self.yolo_model(x)
            if len(self.yolo_feature_tensors) == 0:
                raise RuntimeError("YOLOv5 forward produced no 4D feature maps for backbone extraction.")
            return self._pick_stride_features(self.yolo_feature_tensors, in_h, in_w)

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
