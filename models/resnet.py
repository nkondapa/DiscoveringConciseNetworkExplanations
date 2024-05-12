import torch
from torch import nn
import torchvision.models.resnet as resnet_module
# import networks.modified_resnet as resnet_module


class ResNet(torch.nn.Module):

    def __init__(self, num_classes, base_resnet_model='resnet18', pretrained=True):
        super(ResNet, self).__init__()
        _resnet = resnet_module.__dict__[base_resnet_model](pretrained)
        self.num_classes = num_classes
        self.features = nn.Sequential()
        for name, module in _resnet.named_children():
            if name == 'avgpool':
                break
            self.features.add_module(name, module)

        self.avgpool = _resnet.avgpool
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(_resnet.fc.in_features, num_classes),
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        out = self.classifier(x)

        return out

    def compute_loss(self, output, target):
        return self.loss_fn(output, target)
