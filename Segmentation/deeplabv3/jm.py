import numpy as np
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, SAGEConv


__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def model_size(model):
    total_size = 0
    for param in model.parameters():
        # °¢ ÆÄ¶ó¹ÌÅÍÀÇ ¿ø¼Ò °³¼ö °è»ê
        num_elements = torch.prod(torch.tensor(param.size())).item()
        # ¿ø¼Ò Å¸ÀÔ º°·Î ¹ÙÀÌÆ® Å©±â °è»ê (¿¹: float32 -> 4 bytes)
        num_bytes = num_elements * param.element_size()
        total_size += num_bytes
    return total_size

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class blockSAGEsq(nn.Module):
    def __init__(self,hidden, inner):
        super(blockSAGEsq,self).__init__()
        self.hidden = hidden
        self.inner  = inner
        self.sage1 = SAGEConv(hidden, inner)
        #self.linear = nn.Linear(inner, hidden)
    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.elu(x)
        #x = self.linear(x)
        #x = F.elu(x)

        return x, edge_index
class StrideGraphSAGE(torch.nn.Module):
    def __init__(self, hidden,inner,n_layers,graph_stride,grid_size):
        super(StrideGraphSAGE, self).__init__()

        self.embedding_size = hidden

        self.stride = graph_stride
        self.grid_size = grid_size

        self.gn_layers = nn.ModuleList()

        if n_layers > 8:
            n_layers = 8

        for _ in range(n_layers):
            self.gn_layers.append(blockSAGEsq(hidden,inner))

    def stride_edge(self, grid_size, stride):
        edge_index = []

        for i in range(grid_size):
            for j in range(grid_size):
                current = i * grid_size + j
                # Connect with neighbor 2 steps to the right
                if j < grid_size - stride:
                    edge_index.append([current, current + stride])
                # Connect with neighbor 2 steps below
                if i < grid_size - stride:
                    edge_index.append([current, current + stride * grid_size])
                # Connect with neighbor 2 steps below and to the right (diagonal)
                if j < grid_size - stride and i < grid_size - stride:
                    edge_index.append([current, current + stride * grid_size + stride])
                # Connect with neighbor 2 steps below and to the left (diagonal)
                if j > stride and i < grid_size - stride:
                    edge_index.append([current, current + stride * grid_size - stride])

        edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous()#.cuda()
        return edge_idx

    def feature2graph(self,feature_map,edge_index):

        batch_size, channels, height, width = feature_map.shape

        # list to store graph data for each item in the batch
        data_list = []

        # iterate through the batch
        for i in range(batch_size):
            # extracting the feature_map for this item in the batch
            single_map = feature_map[i]

            # reshaping (100, 3, 3) to (9, 100)
            x = single_map.view(channels, height * width).permute(1, 0)  # x has shape [9, 100]

            # constructing edge_index (in this example, making fully connected graph for simplicity)

            # create Data instance for this item in the batch
            data = Data(x=x, edge_index=edge_index)

            # appending the data to the data_list
            data_list.append(data)

        # create a batch from the data_list
        batch = Batch.from_data_list(data_list)

        return batch

    def graph2feature(self, graph, num_nodes, feature_shape):
        batch_size = graph.size(0) // num_nodes

        # list to store feature maps for each item in the batch
        feature_maps = []

        # iterate through the batch
        for i in range(batch_size):
            # extracting the tensor for this item in the batch
            single_tensor = graph[i * num_nodes: (i + 1) * num_nodes]


            # reshaping (num_nodes, 100) to (256, 3, 3)
            single_map = single_tensor.view(*feature_shape)  # single_map has shape [256, 3, 3]

            # appending the feature map to the feature_maps list
            feature_maps.append(single_map)

        # stack all feature maps to a single tensor
        feature_maps = torch.stack(feature_maps)

        return feature_maps

    def forward(self, x):

        x0 = x

        edge_idx = self.stride_edge(self.grid_size,self.stride)
        x = self.feature2graph(x,edge_idx)


        x_s = [None] * len(self.gn_layers)
        x_s_f = [None] * len(self.gn_layers)

        for ii in range(len(self.gn_layers)):
            x_n, edge = self.gn_layers[ii](x.x, edge_idx)
            x_s[ii] = x_n
            x_s_f[ii] = self.graph2feature(x_n, num_nodes=(self.grid_size ** 2),
                                           feature_shape=(self.embedding_size, 33, 33))

        x_s_f.append(x0)
        output = torch.cat(x_s_f, dim=1)

        return output

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(2048, 512,  kernel_size=3, stride=1,padding=6,dilation=6)
        self.conv2 = nn.Conv2d(512, 64, kernel_size=3, stride=1,padding = 3,dilation=3)

        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(512,momentum=0.0003)
        self.bn2 = nn.BatchNorm2d(64,momentum=0.0003)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)

        return x


class DecoderCNN(nn.Module):
    def __init__(self):
        super(DecoderCNN, self).__init__()
        self.conv3_transpose = nn.ConvTranspose2d(512, 1024, kernel_size=1, stride=1)
        self.conv2_transpose = nn.ConvTranspose2d(1024,2048, kernel_size=2, stride=3,dilation=2)
        self.gelu = nn.GELU()
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(2048)


    def forward(self, x):
        x = self.conv3_transpose(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = self.conv2_transpose(x)
        x = self.gelu(x)

        return x




class PyramidGNN(nn.Module):
    def __init__(self,C,depth,num_classes):
        super(PyramidGNN, self).__init__()
        self.depth = depth
        self.numclasses = num_classes

        hidden_size = depth
        inner_size = depth

        self.encoder = EncoderCNN()
        #self.decoder = DecoderCNN()

        self.grid_size = 33

        self.gnn1 = StrideGraphSAGE(hidden=depth, inner=depth, n_layers=int((self.grid_size-1)/1), graph_stride=2,
                                    grid_size=self.grid_size)
        self.gnn2 = StrideGraphSAGE(hidden=depth, inner=depth, n_layers=int((self.grid_size - 1) / 2), graph_stride=2,
                                    grid_size=self.grid_size)
        self.gnn3 = StrideGraphSAGE(hidden=depth, inner=depth, n_layers=int((self.grid_size - 1) / 3), graph_stride=3,
                                    grid_size=self.grid_size)
        self.gnn4 = StrideGraphSAGE(hidden=depth, inner=depth, n_layers=int((self.grid_size - 1) / 4), graph_stride=4,
                                    grid_size=self.grid_size)
        self.gnn5 = StrideGraphSAGE(hidden=depth, inner=depth, n_layers=int((self.grid_size - 1) / 5), graph_stride=5,
                                    grid_size=self.grid_size)


        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(2816,256, kernel_size=1, stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.0003)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

        #self.bn2 = nn.BatchNorm2d(512, momentum=0.0003)
        #self.conv3 = nn.Conv2d(512,num_classes, kernel_size=1, stride=1)
        #self.bn3 = nn.BatchNorm2d(128, momentum=0.0003)
        #self.conv4 = nn.Conv2d(128,num_classes, kernel_size=1, stride=1)

        #self.upsampling = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=2,stride=3,dilation=2)


    def forward(self, x):
        # Encoder
        x_origin = x

        size = [x_origin.shape[2],x_origin.shape[3]]
        #print(x_origin.shape[2],x_origin.shape[3])

        x0 = self.encoder(x)

        x1 = self.gnn1(x0)
        x2 = self.gnn2(x0)
        x3 = self.gnn3(x0)
        x4 = self.gnn4(x0)
        x5 = self.gnn5(x0)

        #print(x0.shape[1],x1.shape[1],x2.shape[1],x3.shape[1],x4.shape[1],x5.shape[1])
        output = torch.cat([x0,x1,x2,x3,x4,x5], dim=1)
        #print(output.shape)
        # Decoder
        #x = self.decoder(x)

        # Output
        x = self.conv2(output)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        #x = self.bn2(x)
        #x = self.gelu(x)
        #x = self.conv3(x)
       #x = self.bn3(x)
       #x = self.gelu(x)
       #x = self.conv4(x)
       #x = self.gelu(x)
        #x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=21, num_groups=None, weight_std=False, beta=False):
        self.inplanes = 64
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)
        self.conv = Conv2d if weight_std else nn.Conv2d

        super(ResNet, self).__init__()
        if not beta:
            self.conv1 = self.conv(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                self.conv(3, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,dilation=2)
        self.pyramid_gnn = PyramidGNN(512*block.expansion,64, num_classes )
        #self.upsample = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=2, stride=1, padding=1,dilation=2)

        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) #block1
        x = self.layer2(x) #block2
        x = self.layer3(x) #block3
        x = self.layer4(x) #block4

        x = self.pyramid_gnn(x)

        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)


        return x



def resnet50(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    """model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model"""
    if pretrained:
        print("Pretrained!!")
        model_dict = model.state_dict()
        if num_groups and weight_std:
            print("1")
            pretrained_dict = torch.load('data/R-101-GN-WS.pth.tar')
            overlap_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
            assert len(overlap_dict) == 312
        elif not num_groups and not weight_std:
            #print("2")
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
            overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        else:
            print("3")
            raise ValueError('Currently only support BN or GN+WS')
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    else:
        print("Not Pretrained!!")

    return model

def resnet101(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_groups=num_groups, weight_std=weight_std, **kwargs)
    if pretrained:
        print("0")
        model_dict = model.state_dict()
        if num_groups and weight_std:
            print("1")
            pretrained_dict = torch.load('data/R-101-GN-WS.pth.tar')
            overlap_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
            assert len(overlap_dict) == 312
        elif not num_groups and not weight_std:
            print("2")
            pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
            overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        else:
            print("3")
            raise ValueError('Currently only support BN or GN+WS')
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

model=resnet50()
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)#
# print number of parameters#
print(f"Number of parameters: {num_params / (1000.0 ** 2): .3f} M")
from torchsummary import summary

# Print the model summary
summary(model, input_size=(3, 513, 513))  # Replace (3, 224, 224) with your input size
