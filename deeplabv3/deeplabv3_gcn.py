import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torchvision.models import resnet
from typing import List
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected


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
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class CustomGraphSAGE(torch.nn.Module):
    def __init__(self, size):
        super(CustomGraphSAGE, self).__init__()

        self.sage1 = GCNConv(size, size)
        self.sage2 = GCNConv(size, size)
        self.sage3 = GCNConv(size, size)

    def forward(self, x, edge_index):
        print("CustomGraphSAGE input:", x.size(), edge_index.size())
        x = self.sage1(x, edge_index)

        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.sage2(x, edge_index)

        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.sage3(x, edge_index)

        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super(GCN, self).__init__()
        self.gcn_conv = CustomGraphSAGE(in_channels)
        self.edge_index = self.create_edge_index(16, 16)

    def create_edge_index(self, height, width):
        row_indices = []
        col_indices = []

        for i in range(height):
            for j in range(width):
                current_index = i * width + j

                # Add self-loop connection
                row_indices.append(current_index)
                col_indices.append(current_index)

                # Add connections to the right and bottom nodes if they exist
                if j < width - 1:
                    right_index = i * width + j + 1
                    row_indices.append(current_index)
                    col_indices.append(right_index)
                if i < height - 1:
                    bottom_index = (i + 1) * width + j
                    row_indices.append(current_index)
                    col_indices.append(bottom_index)

        edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)

        return edge_index

    def feature2graph(self, feature_map, edge_index):
        batch_size, channels, height, width = feature_map.shape

        # List to store graph data for each item in the batch
        data_list = []

        # Iterate through the batch
        for i in range(batch_size):
            # Extract the feature_map for this item in the batch
            single_map = feature_map[i]

            # Reshape feature_map to match the graph size
            x = single_map.view(channels, height * width).permute(1, 0)  # x has shape [height * width, channels]

            # Create Data instance for this item in the batch
            data = Data(x=x, edge_index=edge_index)

            # Appending the data to the data_list
            data_list.append(data)

        # Create a batch from the data_list
        batch = Batch.from_data_list(data_list)

        return batch

    def forward(self, x):
        print('GCN input:', x.size())

        # Pass the input and edge_index to the GCN convolution layer
        x = self.feature2graph(x, self.edge_index)
        x = self.gcn_conv(x.x, x.edge_index)

        # Reshape the output tensor back to the original size
        output_tensor = x.view(2, x.size(1),  16, 16)
        print('GCN output:', output_tensor.size())

        return output_tensor







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

        ## add gcn code in here

        modules.append(GCN(in_channels,out_channels,64))



        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("****ASPP****")
        print(x.size())
        res = []
        for conv in self.convs:
            print('conv',conv, conv(x).size())
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        x = self.project(res)
        print('after aspp output ', x.size())
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


class DeepLabv3(nn.Module):
    def __init__(self, backbone, classifier, fcn):
        super(DeepLabv3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.fcn = fcn

    def forward(self, x):
        print('back bone input',x.size())
        features = self.backbone(x)
        print('classifiaction input',features.size())
        output1 = self.classifier(features)
        output2 = self.fcn(output1)
        return output2


def model_load():
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

    classifier = DeepLabHead(2048, num_classes)
    fcn = FCNHead(1024, num_classes)

    # Create an instance of the combined model
    model = DeepLabv3(backbone, classifier, fcn)
    return model


from torchsummary import summary

# Assuming you have instantiated and loaded your model
model = model_load()
print(model)
# Define input shapes for each input to your model
input_shapes = [(3, 512, 512)]  # Example input shapes, replace with actual shapes

# Use torchsummary to get the model summary
summary(model, input_shapes)
