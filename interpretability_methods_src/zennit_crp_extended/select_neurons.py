import os
import numpy as np
from interpretability_methods_src.zennit_crp_extended.utils import *


def select_top_k_neurons(method, folder_name, dataset, class_subset, topk=250):
    relevances = load_relevances(folder_name)['layer_relevances_dict']

    folder_name = f'{folder_name}/top_k_neurons/{method}/'
    os.makedirs(folder_name, exist_ok=True)

    class_subset_indices_dict = {}
    for cs in class_subset:
        class_subset_indices_dict[cs] = np.array(dataset.get_class_subset_indices(cs))


    layer_topk_dict = {}
    if 'global' == method:
        tmp = select_global_topk_neurons(dataset=dataset, class_subset=class_subset,
                                         class_subset_indices_dict=class_subset_indices_dict,
                                         layer_relevances_dict=relevances,
                                         top_k=topk
                                         )
    elif 'average' == method:
        # layer_topk_dict['average'] = {}
        tmp = select_topk_neurons_across_subset(dataset=dataset, class_subset=class_subset,
                                                class_subset_indices_dict=class_subset_indices_dict,
                                                layer_relevances_dict=relevances,
                                                top_k=topk,
                                                mode='average'
                                                )

    elif 'max' == method:
        tmp = select_topk_neurons_across_subset(dataset=dataset, class_subset=class_subset,
                                                class_subset_indices_dict=class_subset_indices_dict,
                                                layer_relevances_dict=relevances,
                                                top_k=topk,
                                                mode='max'
                                                )

    else:
        tmp = {}
        for cs in class_subset:
            tmp[cs] = {}
            for index in class_subset_indices_dict[cs]:
                image_path = dataset.samples[index][0]
                tmp[cs][image_path] = {}
                for layer in layer_topk_dict:
                    tmp[cs][image_path][layer] = {
                        'channels': (-relevances[layer][index]).argsort()[
                                    :layer_topk_dict[layer]].tolist()}

    with open(os.path.join(folder_name, f'topk={topk}.pkl'), 'wb') as f:
        pkl.dump(tmp, f)

    return tmp


def select_global_topk_neurons(dataset, class_subset, class_subset_indices_dict, layer_relevances_dict, top_k):
    """
    Helper function to flatten the layer_attribution_dict
    :param layer_attribution_dict: the layer_attribution_dict
    :return: the flattened layer_attribution_dict
    """

    tmp = {}
    for cs in class_subset:
        class_subset_indices = class_subset_indices_dict[cs]
        tmp_layer_names = []
        tmp_values = []
        tmp_value_true_indices = []
        for layer in layer_relevances_dict:
            tmp_layer_names.append(np.array([layer] * layer_relevances_dict[layer].shape[1]))
            tmp_values.append(torch.FloatTensor(layer_relevances_dict[layer][class_subset_indices]))
            tmp_value_true_indices.append(torch.arange(layer_relevances_dict[layer].shape[1]))

        tmp_layer_names = np.concatenate(tmp_layer_names)
        tmp_values = torch.cat(tmp_values, dim=1)
        tmp_value_true_indices = torch.cat(tmp_value_true_indices, dim=0)
        top_k_indices = tmp_values.argsort(dim=1, descending=True)[:, :top_k]
        tmp[cs] = {}
        for ii, index in enumerate(class_subset_indices_dict[cs]):
            top_k_image_indices = top_k_indices[ii]
            top_k_image_layer_names = tmp_layer_names[top_k_image_indices]
            image_path = dataset.samples[index][0]
            tmp[cs][image_path] = {}
            for layer in np.unique(top_k_image_layer_names):
                tmp[cs][image_path][layer] = {
                    'channels': (
                        tmp_value_true_indices[top_k_image_indices[top_k_image_layer_names == layer]]).tolist()}
    return tmp


def select_topk_neurons_across_subset(dataset, class_subset, class_subset_indices_dict, layer_relevances_dict,
                                      top_k, mode='average'):
    """
    Helper function to flatten the layer_attribution_dict
    :param layer_attribution_dict: the layer_attribution_dict
    :return: the flattened layer_attribution_dict
    """

    tmp = {}
    for cs in class_subset:
        class_subset_indices = class_subset_indices_dict[cs]
        tmp_layer_names = []
        tmp_values = []
        tmp_value_true_indices = []
        for layer in layer_relevances_dict:
            tmp_layer_names.append(np.array([layer] * layer_relevances_dict[layer].shape[1]))
            tmp_values.append(torch.FloatTensor(layer_relevances_dict[layer][class_subset_indices]))
            tmp_value_true_indices.append(torch.arange(layer_relevances_dict[layer].shape[1]))

        tmp_layer_names = np.concatenate(tmp_layer_names)
        tmp_values = torch.cat(tmp_values, dim=1)
        tmp_value_true_indices = torch.cat(tmp_value_true_indices, dim=0)
        if mode == 'average':
            top_k_indices = tmp_values.mean(0).argsort(descending=True)[:top_k]
        elif mode == 'max':
            top_k_indices = tmp_values.max(0).argsort(descending=True)[:top_k]
        else:
            raise NotImplementedError
        top_k_indices = top_k_indices.repeat(tmp_values.shape[0], 1)
        tmp[cs] = {}
        for ii, index in enumerate(class_subset_indices_dict[cs]):
            top_k_image_indices = top_k_indices[ii]
            top_k_image_layer_names = tmp_layer_names[top_k_image_indices]
            image_path = dataset.samples[index][0]
            tmp[cs][image_path] = {}
            for layer in np.unique(top_k_image_layer_names):
                tmp[cs][image_path][layer] = {
                    'channels': (
                        tmp_value_true_indices[top_k_image_indices[top_k_image_layer_names == layer]]).tolist()}
    return tmp
