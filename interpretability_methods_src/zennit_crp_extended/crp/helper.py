import torch
from typing import List

def get_layer_names(model: torch.nn.Module, types: List):
    """
    Retrieves the layer names of all layers that belong to a torch.nn.Module type defined
    in 'types'.

    Parameters
    ----------
    model: torch.nn.Module
    types: list of torch.nn.Module
        Layer types i.e. torch.nn.Conv2D

    Returns
    -------
    layer_names: list of strings


    """

    layer_names = []
    layers = []
    for name, layer in model.named_modules():
        for layer_definition in types:
            if isinstance(layer, layer_definition) or issubclass(layer.__class__, layer_definition):
                if name not in layer_names:
                    layer_names.append(name)
                    layers.append(layer)

    return layer_names, layers
