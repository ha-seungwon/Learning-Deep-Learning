from torchvision.models import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional



        

class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)

        
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )
class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        x = self.project(res)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        # Combine the components into a single model
class DeepLabv3(nn.Module):
    def __init__(self, backbone, classifier, fcn):
        super(DeepLabv3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.fcn = fcn
    
    def forward(self, x):
        features = self.backbone(x)
        output1 = self.classifier(features)
        print(output1.size())
        output2 = self.fcn(output1)
        return output2


def model_load():
    num_classes=21
    resnet_backbone = resnet.resnet50(pretrained=True)
    backbone=nn.Sequential(
            resnet_backbone.conv1,
            resnet_backbone.bn1,
            resnet_backbone.relu,
            resnet_backbone.maxpool,
            resnet_backbone.layer1,
            resnet_backbone.layer2,
            resnet_backbone.layer3,
            resnet_backbone.layer4
            )
    
    classifier=DeepLabHead(2048,1024)
    fcn=FCNHead(1024, num_classes)


    # Create an instance of the combined model
    model = DeepLabv3(backbone, classifier, fcn)
    return model

