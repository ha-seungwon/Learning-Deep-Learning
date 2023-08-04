import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torchvision.models import resnet
from typing import List
from torch_geometric.data import Data, Batch
import warnings
import torch.utils.model_zoo as model_zoo

warnings.filterwarnings("ignore")

__all__ = ['ResNet', 'resnet50','resnet101']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)



    return model

def resnet50(progress=True, **kwargs):
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], progress,
                    **kwargs)
    model_dict = model.state_dict()
    pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(overlap_dict)
    model.load_state_dict(model_dict)

    return model

def resnet101(progress=True, **kwargs):

    model=_resnet('resnet101', Bottleneck, [3, 4, 23, 3], progress,
                   **kwargs)
    model_dict = model.state_dict()
    pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
    overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(overlap_dict)
    model.load_state_dict(model_dict)

    return model

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
        in_channels = in_channels // 2
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

    def forward(self, x, edge_index,edge_idx_1_1):
        x = self.sage1(x, edge_index)
        x = F.dropout(x)
        x = F.relu(x)
        x = self.sage2(x, edge_idx_1_1)
        x = F.dropout(x)
        x = F.relu(x)
        x = self.sage3(x, edge_index)
        return x

class GCN(torch.nn.Module):
    def __init__(self, model_type: str, in_channels: int, out_channels: int, gcn_rate: int, grid_size: int):
        super(GCN, self).__init__()
        if model_type == 'sage':
            self.gcn_conv = GCN_Layer_sage(in_channels, out_channels)
        else:
            self.gcn_conv = GCN_Layer(in_channels, out_channels)
        self.grid_size = grid_size
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
                if j > stride and i < grid_size - stride:
                    edge_index.append([current, current + grid_size * stride - stride])
        edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous()#.cuda()
        return edge_idx
    
    def edge1_1(self,grid_size):
        edge_index = []
        for i in range(grid_size):
            for j in range(grid_size):
                current = i * grid_size + j

                # Connect to the right neighbor
                if j < grid_size - 1:
                    edge_index.append([current, current + 1])

                # Connect to the bottom neighbor
                if i < grid_size - 1:
                    edge_index.append([current, current + grid_size])

        edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # .cuda()
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
        edge_idx_1_1=self.edge1_1(self.grid_size)


        x_1 = self.feature2graph(x, edge_idx_1_1)
        x = self.feature2graph(x, edge_idx)
        
        x = self.gcn_conv(x.x, x.edge_index,x_1.edge_index)
        x = self.graph2feature(x, num_nodes=(self.grid_size ** 2))
        return x

class DeepLabHead(nn.Sequential):
    def __init__(self, gcn_model_type: str, in_channels: int, num_classes: int, grid_size: int, atrous_rates: List[int],
                 gcn_rates: List[int]):
        super().__init__(
            ASPP(gcn_model_type, in_channels, grid_size, atrous_rates, gcn_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, dilation=1)

        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(256, momentum=0.0003)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        return x

class ASPP(nn.Module):
    def __init__(self, gcn_model_type: str, in_channels: int, grid_size: int, atrous_rates: List[int],
                 gcn_rates: List[int],
                 out_channels: int = 256,momentum: int =0.0003):
        super().__init__()

        modules = []
        gcn_modules = []
        #modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #                             nn.BatchNorm2d(out_channels,momentum), nn.ReLU(inplace=True)))

        rates = tuple(atrous_rates)
        gcn_rates = tuple(gcn_rates)
        if len(rates) != 0:
            for rate in rates:
                modules.append(ASPPConv(in_channels, out_channels, rate))

        self.reduce_channel = EncoderCNN()

        if len(gcn_rates) != 0:
            for gcn_rate in gcn_rates:
                gcn_modules.append(GCN(gcn_model_type, in_channels // 8, out_channels, gcn_rate, grid_size))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.gcn_convs = nn.ModuleList(gcn_modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs + self.gcn_convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels,momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        x = self.reduce_channel(x)
        for gcn_conv in self.gcn_convs:
            res.append(gcn_conv(x))

        for i in res:
            print(i.size())
        res = torch.cat(res, dim=1)
        x = self.project(res)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int,momentum=0.0003):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels,momentum),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,momentum=0.0003):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels,momentum),
            nn.ReLU(inplace=True),
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

def model_load(backbone_arch,gcn_model_type: str, atrous_rates: List[int], gcn_rates: List[int]):
    num_classes = 21
    if backbone_arch=='resnet50':
        resnet_backbone = resnet50()
    elif backbone_arch=='resnet101':
        resnet_backbone=resnet101()
    

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

    classifier = DeepLabHead(gcn_model_type, 2048, num_classes, 33, atrous_rates, gcn_rates)
    fcn = FCNHead(21, num_classes)

    model = DeepLabv3(backbone, classifier, fcn)
    #model = model.cuda()

    return model



model=model_load('resnet50','sage',[1,3],[4,8,16])
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)#
# print number of parameters#
print(f"Number of parameters: {num_params / (1000.0 ** 2): .3f} M")
from torchsummary import summary

# Print the model summary
summary(model, input_size=(3, 513, 513))  # Replace (3, 224, 224) with your input size
