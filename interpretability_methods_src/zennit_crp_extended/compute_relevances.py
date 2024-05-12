import os
from tqdm import tqdm
import torch
from interpretability_methods_src.zennit_crp.crp.attribution import CondAttribution
from interpretability_methods_src.zennit_crp_extended.crp.concepts import ChannelConcept
import pickle as pkl


def compute_relevances(model, dataset, composite, layer_names, folder_name):
    bsz = 1
    loader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=False)

    # define LRP rules and canonizers in zennit
    # composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])

    # load interpretability_methods.zennit_crp.semantic_crp toolbox
    attribution = CondAttribution(model)

    # here, each channel is defined as a concept
    # or define your own notion!
    attribute_function = ChannelConcept.attribute_unreduced
    # get layer names of Conv2D and MLP layers
    layer_names = layer_names

    layer_relevances_dict = {}
    layer_activations_dict = {}

    labels = []
    predictions = []

    folder_name = f'{folder_name}/relevances/'
    os.makedirs(folder_name, exist_ok=True)

    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    for idx, (img, label, im_path) in enumerate(loader):
        pbar.update(1)
        img = img.to('cuda')
        label = label.to('cuda')
        img.requires_grad_(True)
        conditions = [{'y': label.item()}]

        attr = attribution(img, conditions, composite, record_layer=layer_names)

        labels.append(label.item())
        predictions.append(attr.prediction.argmax().item())

        for tln in layer_names:
            if tln not in layer_relevances_dict:
                layer_relevances_dict[tln] = torch.zeros(
                    size=(len(loader), attr.relevances[tln].shape[1]), dtype=torch.float32)
                layer_activations_dict[tln] = torch.zeros(
                    size=(len(loader), attr.activations[tln].shape[1]), dtype=torch.float32)

            relevance, normed_relevance, abs_norm = attribute_function(attr.relevances[tln])

            layer_activations_dict[tln][idx] = attr.activations[tln].sum((-1, -2)).cpu().detach()
            layer_relevances_dict[tln][idx] = relevance.sum((-1, -2)).cpu().detach()

    labels = torch.Tensor(labels)
    predictions = torch.Tensor(predictions)

    with open(folder_name + f'/layer_relevances_dict.pkl', 'wb') as f:
        pkl.dump(layer_relevances_dict, f)

    with open(folder_name + f'/layer_activations_dict.pkl', 'wb') as f:
        pkl.dump(layer_activations_dict, f)

    torch.save(labels, folder_name + '/labels.pt')
    torch.save(predictions, folder_name + '/predictions.pt')


def load_relevances(folder_name):
    folder_name = f'{folder_name}/relevances/'

    try:
        with open(folder_name + f'/layer_relevances_dict.pkl', 'rb') as f:
            layer_relevances_dict = pkl.load(f)
        with open(folder_name + f'/layer_activations_dict.pkl', 'rb') as f:
            layer_activations_dict = pkl.load(f)
        labels = torch.load(folder_name + '/labels.pt')
        predictions = torch.load(folder_name + '/predictions.pt')
        return True
    except FileNotFoundError:
        print("Relevances not found in the specified folder.")
        return None


def check_relevances(folder_name):
    folder_name = f'{folder_name}/relevances/'

    try:
        if os.path.exists(folder_name + f'/layer_relevances_dict.pkl') and os.path.exists(
                folder_name + f'/layer_activations_dict.pkl') and os.path.exists(
                folder_name + '/labels.pt') and os.path.exists(folder_name + '/predictions.pt'):
            return True
        else:
            return False
    except FileNotFoundError:
        print("Relevances not found in the specified folder.")
        return False
