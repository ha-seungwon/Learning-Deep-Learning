import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torchvision.models import resnet
from typing import List
from torch_geometric.data import Data, Batch

import warnings
warnings.filterwarnings("ignore")


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


class GCN_Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_Layer, self).__init__()
        self.sage1 = GCNConv(in_channels, in_channels)
        self.sage2 = GCNConv(in_channels, in_channels)
        self.sage3 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.dropout(x)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        x = F.dropout(x)
        x = F.relu(x)
        x = self.sage3(x, edge_index)
        return x


class GCN_Layer_sage(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_Layer_sage, self).__init__()
        self.sage1 = SAGEConv(in_channels, in_channels)
        self.sage2 = SAGEConv(in_channels, in_channels)
        self.sage3 = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.dropout(x)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        x = F.dropout(x)
        x = F.relu(x)
        x = self.sage3(x, edge_index)
        return x


class GCN(torch.nn.Module):
    def __init__(self, model_type: str, in_channels: int, out_channels: int, gcn_rate: int):
        super(GCN, self).__init__()
        if model_type == 'sage':
            self.gcn_conv = GCN_Layer_sage(in_channels, out_channels)
        else:
            self.gcn_conv = GCN_Layer(in_channels, out_channels)
        self.grid_size = 17
        self.stride = gcn_rate

    def edge(self, grid_size, stride):
        edge_index = []
        for i in range(0, grid_size, stride):
            for j in range(0, grid_size, stride):
                current = i * grid_size + j
                if j < grid_size - stride:
                    edge_index.append([current, current + stride])
                if i < grid_size - stride:
                    edge_index.append([current, current + grid_size * stride])
                if j < grid_size - stride and i < grid_size - stride:
                    edge_index.append([current, current + grid_size * stride + stride])
                if j > 0 and i < grid_size - stride:
                    edge_index.append([current, current + grid_size * stride - stride])
        edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_idx

    def feature2graph(self, feature_map, edge_index):
        batch_size, channels, height, width = feature_map.shape
        data_list = []
        for i in range(batch_size):
            single_map = feature_map[i]
            x = single_map.view(channels, height * width).permute(1, 0)
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        return batch

    def graph2feature(self, graph, num_nodes):
        batch_size = graph.size(0) // num_nodes
        channels = graph.size(1)
        feature_shape = (channels, self.grid_size, self.grid_size)
        feature_maps = []
        for i in range(batch_size):
            single_tensor = graph[i * num_nodes: (i + 1) * num_nodes]
            single_map = single_tensor.view(feature_shape)
            feature_maps.append(single_map)
        feature_maps = torch.stack(feature_maps)
        return feature_maps

    def forward(self, x):
        edge_idx = self.edge(self.grid_size, self.stride)
        x = self.feature2graph(x, edge_idx)
        x = self.gcn_conv(x.x, x.edge_index)
        x = self.graph2feature(x, num_nodes=(self.grid_size ** 2))
        return x


class DeepLabHead(nn.Sequential):
    def __init__(self, gcn_model_type: str, in_channels: int, num_classes: int, atrous_rates: List[int],
                 gcn_rates: List[int]):
        super().__init__(
            ASPP(gcn_model_type, in_channels, atrous_rates, gcn_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPP(nn.Module):
    def __init__(self, gcn_model_type: str, in_channels: int, atrous_rates: List[int], gcn_rates: List[int],
                 out_channels: int = 256):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU()))

        rates = tuple(atrous_rates)
        gcn_rates = tuple(gcn_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        for gcn_rate in gcn_rates:
            modules.append(GCN(gcn_model_type, in_channels, out_channels, gcn_rate))

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
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
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


class DeepLabv3(nn.Module):
    def __init__(self, backbone, classifier, fcn):
        super(DeepLabv3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.fcn = fcn

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        features = self.backbone(x)
        output1 = self.classifier(features)
        output2 = self.fcn(output1)

        model_output = nn.Upsample(size, mode='bilinear', align_corners=True)(output2)
        return model_output


def model_load(gcn_model_type: str, atrous_rates: List[int], gcn_rates: List[int]):
    num_classes = 21
    resnet_backbone = resnet.resnet50(pretrained=True)
    backbone = nn.Sequential(
        resnet_backbone.conv1,
        resnet_backbone.bn1,
        resnet_backbone.relu,
        resnet_backbone.maxpool,
        resnet_backbone.layer1,
        resnet_backbone.layer2,
        resnet_backbone.layer3,
        resnet_backbone.layer4,
    )

    classifier = DeepLabHead(gcn_model_type, 2048, num_classes, atrous_rates, gcn_rates)
    fcn = FCNHead(21, num_classes)

    model = DeepLabv3(backbone, classifier, fcn)
    return model


from torchsummary import summary

model = model_load('sage',[3],[2])

input_shapes = [(3, 513, 513)]
summary(model, input_shapes)
