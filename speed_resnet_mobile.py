import torch
import time
import numpy as np
import torchvision

from utils.common import get_model, merge_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.device('cuda:0')
print(device)

torch.backends.cudnn.benchmark = True

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# model = torchvision.models.resnet34(pretrained=True)
# model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights)
# model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
# model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
# model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)

model.cuda()
x = torch.ones((1, 3, 64, 64)).cuda()
for i in range(10):
    with torch.no_grad():
        y = model(x)
        # y = model.features(x)

t_all = []
for i in range(500):
    t1 = time.time()
    with torch.no_grad():
        y = model(x)
        # y = model.features(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1 * 1000, 'ms')
print('average fps:', 1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1 * 1000, 'ms')
print('fastest fps:', 1 / min(t_all))

print('slowest time:', max(t_all) / 1 * 1000, 'ms')
print('slowest fps:', 1 / max(t_all))
