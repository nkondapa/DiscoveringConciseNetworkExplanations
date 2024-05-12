import pickle as pkl
import torch
from interpretability_methods_src.zennit_crp_extended.crp.graph import trace_model_graph


def load_relevances(folder_name):
    folder_name = f'{folder_name}/relevances/'
    with open(folder_name + f'/layer_relevances_dict.pkl', 'rb') as f:
        layer_relevances_dict = pkl.load(f)
    with open(folder_name + f'/layer_activations_dict.pkl', 'rb') as f:
        layer_activations_dict = pkl.load(f)

    labels = torch.load(folder_name + '/labels.pt')
    predictions = torch.load(folder_name + '/predictions.pt')

    layer_attribution_labels = labels
    layer_attribution_predictions = predictions
    shapes = {layer: layer_relevances_dict[layer].shape for layer in layer_relevances_dict}

    out_dict = dict(layer_activations_dict=layer_activations_dict, layer_relevances_dict=layer_relevances_dict)
    return out_dict


def compute_skip_conditions(layer_names, model, dataset, folder_name, mode=None):

    try:
        with open(folder_name + f'/skip_conditions.pkl', 'rb') as f:
            return pkl.load(f)
    except FileNotFoundError:
        pass

    model_graph = trace_model_graph(model, (1, 3, *dataset.default_image_resize), layer_names)
    print('Computing skip conditions...')
    skip_conditions = {}
    if mode is None:
        for layer in layer_names:
            conditions = [{layer: [0]}]
            conditions = model_graph.exclude_parallel_layers(conditions)
            for key in list(conditions[0].keys()):
                if key == layer:
                    del conditions[0][key]
            skip_conditions[layer] = conditions
    elif mode == 'explicit':
        # TODO add check for model type -- right now assume resnet152
        skip_conditions = {}
        first_pass = {}
        for layer in layer_names:
            if 'layer' in layer:
                l_name = '.'.join(layer.split('.')[:-1])
                if 'downsample' in l_name:
                    l_name = l_name.replace('.downsample', '')

                if l_name not in first_pass:
                    val = 'downsample' in layer
                    first_pass[l_name] = [val, [layer]]
                else:
                    val = 'downsample' in layer
                    first_pass[l_name][1].append(layer)
                    first_pass[l_name][0] = (first_pass[l_name][0] or val)
            else:
                first_pass[layer] = [False, []]

        for layer in layer_names:
            if 'layer' in layer:
                l_name = '.'.join(layer.split('.')[:-1])
                if 'downsample' in l_name:
                    l_name = l_name.replace('.downsample', '')

                if first_pass[l_name][0]:
                    if 'downsample' in layer:
                        cond_dict = {}
                        for lns in first_pass[l_name][1]:
                            if 'downsample' not in lns:
                                cond_dict[lns] = []
                        skip_conditions[layer] = [cond_dict]
                    else:
                        skip_conditions[layer] = [{f'{l_name}.downsample.0': []}]
                else:
                    skip_conditions[layer] = [{}]
            else:
                skip_conditions[layer] = [{}]
    else:
        for layer in layer_names:
            skip_conditions[layer] = [{}]

    with open(folder_name + f'/skip_conditions.pkl', 'wb') as f:
        pkl.dump(skip_conditions, f)

    return skip_conditions


def make_axes_invisible(axes):
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])




