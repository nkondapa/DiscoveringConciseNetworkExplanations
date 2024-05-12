import os
from PIL import Image
import torch
# from interpretability_methods_src.zennit_crp.crp.attribution import CondAttribution
from interpretability_methods_src.zennit_crp_extended.crp.attribution import CondAttribution
from interpretability_methods_src.zennit_crp.crp import image
import matplotlib.pyplot as plt
import os


def save_conditional_attributions_for_dataset(dataset, layer_names, model, composite,
                           selected_neurons, folder_name, topk, skip_conditions=None, num_images_per_class=None, copy_from_topk=None):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    num_images_seen_per_class = {}
    for idx, (img, label, img_path) in enumerate(dataloader):
        print(img_path)
        num_images_seen_per_class[label.item()] = num_images_seen_per_class.get(label.item(), 0) + 1
        if (num_images_per_class is None) or (num_images_seen_per_class[label.item()] <= num_images_per_class):
            img = img.to('cuda')
            img.requires_grad_(True)
            label = label.item()
            save_conditional_attribution(img=img, label=label, img_path=img_path[0], dataset=dataset, model=model,
                               neuron_dict=selected_neurons[label][img_path[0]], composite=composite, folder_name=folder_name,
                               layer_names=layer_names, topk=topk, skip_conditions=skip_conditions, copy_from_topk=copy_from_topk)


def save_conditional_attribution(img, label, img_path, dataset, model, neuron_dict, composite,
                       layer_names, folder_name, topk, skip_conditions=None, copy_from_topk=None):

    if copy_from_topk is not None:
        copy_cond_attribution(img, label, img_path, dataset, model, neuron_dict, composite,
                       layer_names, folder_name, topk, skip_conditions, copy_from_topk)
        return

    class_name = dataset.class_names[label]

    if skip_conditions is None:
        print('No skip condition, overwriting with_skip_conditions to False')
        with_skip_conditions = False
    else:
        with_skip_conditions = True

    folder_name = os.path.join(f'{folder_name}/zennit_crp_{topk}/{class_name}/{img_path.split("/")[-1].split(".")[0]}/')

    os.makedirs(folder_name, exist_ok=True)
    # _image = Image.open(img_path)
    # _image = _image.resize((dataset.default_image_resize[0], dataset.default_image_resize[1]))
    # _image.save(os.path.join(folder_name, 'original.png'))

    skinny_neuron_dict = {}
    for key in neuron_dict:
        if len(neuron_dict[key]['channels']) > 0:
            skinny_neuron_dict[key] = {}
            if isinstance(neuron_dict[key]['channels'], torch.Tensor):
                # TODO only happens in the case when accuracy is not computed,
                #  make dict types consistent across methods
                skinny_neuron_dict[key]['channels'] = neuron_dict[key]['channels'].tolist()
            else:
                skinny_neuron_dict[key] = neuron_dict[key]

    attribution = CondAttribution(model)
    for layer in neuron_dict:
        assert layer in layer_names
        if with_skip_conditions:
            conditions = []
            for cid in neuron_dict[layer]['channels']:

                tmp = {layer: [cid], 'y': [label]}
                for skip_cond in skip_conditions[layer]:
                    tmp.update(skip_cond)
                conditions.append(tmp)
        else:
            conditions = []
            for cid in neuron_dict[layer]['channels']:
                conditions.append({layer: [cid], 'y': [label]})

        if len(conditions) == 0:
            continue

        bsz = min(len(conditions), 1)
        attr_gen = attribution.generate(img, conditions, composite, record_layer=layer_names, batch_size=bsz)

        count = 0
        for k, attr in enumerate(attr_gen):
            heatmap = attr.heatmap.cpu().numpy()

            for i, hm in enumerate(heatmap):
                image.imgify(hm).save(
                    os.path.join(folder_name, f'{layer}_{neuron_dict[layer]["channels"][count]}.png'))
                count += 1

def copy_cond_attribution(img, label, img_path, dataset, model, neuron_dict, composite,
                       layer_names, folder_name, topk, skip_conditions=None, copy_from_topk=None):
    class_name = dataset.class_names[label]
    src_folder_name = os.path.join(f'{folder_name}/zennit_crp_{copy_from_topk}/{class_name}/{img_path.split("/")[-1].split(".")[0]}/')
    folder_name = os.path.join(f'{folder_name}/zennit_crp_{topk}/{class_name}/{img_path.split("/")[-1].split(".")[0]}/')
    os.makedirs(folder_name, exist_ok=True)

    skinny_neuron_dict = {}
    for key in neuron_dict:
        if len(neuron_dict[key]['channels']) > 0:
            skinny_neuron_dict[key] = {}
            if isinstance(neuron_dict[key]['channels'], torch.Tensor):
                # TODO only happens in the case when accuracy is not computed,
                #  make dict types consistent across methods
                skinny_neuron_dict[key]['channels'] = neuron_dict[key]['channels'].tolist()
            else:
                skinny_neuron_dict[key] = neuron_dict[key]

    for layer in neuron_dict:
        assert layer in layer_names

        count = 0
        for k, ch in enumerate(neuron_dict[layer]['channels']):
            src_path = os.path.join(src_folder_name, f'{layer}_{ch}.png')
            dst_path = os.path.join(folder_name, f'{layer}_{ch}.png')
            os.symlink(src_path, dst_path)
            count += 1