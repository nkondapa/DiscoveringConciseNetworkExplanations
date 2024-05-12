import torch
import torchvision

model_opts = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101,
}

model_weights = {
    'resnet18': torchvision.models.ResNet18_Weights,
    'resnet34': torchvision.models.ResNet34_Weights,
    'resnet50': torchvision.models.ResNet50_Weights,
    'resnet101': torchvision.models.ResNet101_Weights,
}


def construct_model(model_type, pretrained=False):
    weights = None
    if pretrained:
        weights = model_weights[model_type]
    model = model_opts[model_type](weights=weights)
    return model

def construct_old_model(model_type, pretrained=False):
    from interpretability_methods_src.zennit_crp_extended.old_model_stuff.networks.resnet import ResNet
    model = ResNet(num_classes=200, base_resnet_model=model_type, pretrained=pretrained)
    return model


def modify_model_output_layer(model, num_classes):
    if 'torchvision.models.resnet' in str(type(model)):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model
