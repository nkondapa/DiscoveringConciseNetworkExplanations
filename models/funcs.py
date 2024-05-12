import os
import torch
from models.resnet import ResNet
from models.resnet import resnet_module
from torchvision import transforms
import torchsummary


def transfer_to_canonical_resnet34(model):
    base = resnet_module.resnet34()
    feature_params = dict(model.features.module.named_modules())
    # feature_params = dict(model.features.named_modules())
    for name, module in base.named_children():
        if name in feature_params:
            setattr(base, name, feature_params[name])
        else:
            print('Not found: {}'.format(name))
    base.avgpool = model.avgpool
    base.fc = model.classifier
    return base


def load_expert_model(model_name, model_path, gpu=None, transfer_to_canonical=True):
    if model_name == 'resnet34_CUB_expert':
        model = _load_expert_model(model_name, model_path, num_classes=200, architecture='resnet34', gpu=gpu, pretrained=False)
        if transfer_to_canonical:
            model = transfer_to_canonical_resnet34(model).cuda().eval()
        return model
    else:
        raise ValueError('Expert model not found')


def _load_expert_model(model_name, model_path, num_classes, architecture, gpu=None, pretrained=False, skip_dataparallel=False):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_name))
        if gpu is None:
            checkpoint = torch.load(model_path)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(model_path, map_location=loc)

        model = load_model(architecture, num_classes, pretrained=False)

        # @ TODO if using distributed training will need to add more code from training script,
        # DataParallel modifies the model structure to be able to handle multiple GPUs
        # DataParallel will divide and allocate batch_size to all available GPUs
        if architecture.startswith('alexnet') or architecture.startswith('vgg') or architecture.startswith('resnet') \
                or architecture.startswith('densenet') or architecture.startswith(
            'efficientnet') or architecture.startswith('convnext'):
            if architecture.startswith('convnext'):
                model.cuda()
            else:
                if skip_dataparallel:
                    model = model.cuda()
                else:
                    model.features = torch.nn.DataParallel(model.features).cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_name, checkpoint['epoch']))

        return model
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

def load_dataset_transforms(dataset_name, eval=False):

    if dataset_name == 'CUB':
        if not eval:
            transform = transforms.Compose([transforms.RandomResizedCrop(448),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.48173460364341736, 0.49579519033432007, 0.43105146288871765],
                                                                 [0.22944197058677673, 0.22590851783752441, 0.26358509063720703])])
        else:
            transform = transforms.Compose([transforms.Resize((448, 448)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.48173460364341736, 0.49579519033432007, 0.43105146288871765],
                                                                 [0.22944197058677673, 0.22590851783752441, 0.26358509063720703])])
            return transform

    else:
        raise ValueError('Dataset not supported')


def get_optimizer(model, architecture, dataset_name, lr_scheduler=None, lr_scheduler_params=None):
    if 'resnet' in architecture and (dataset_name == 'CUB' or dataset_name == 'CUB_224_crop'):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if lr_scheduler == 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
            lambda x: (1 - x / (lr_scheduler_params['max_epochs'] * lr_scheduler_params['dataloader_length'] + 1)) ** 0.9
        ])

    return optimizer, scheduler


def load_model(model_type, num_classes, pretrained=True):
    if 'resnet' in model_type:
        return ResNet(num_classes, base_resnet_model=model_type, pretrained=pretrained)
    else:
        raise ValueError(f'Model {model_type} not supported')

