import argparse
import os
import torch
from interpretability_methods_src.pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
from interpretability_methods_src.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2
from datasets.cub import CUBDataset
from interpretability_methods_src.pytorch_grad_cam import (
    GradCAM, HiResCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, AblationCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad
)
from models.funcs import load_expert_model, load_dataset_transforms
import json


pytorch_gradcam_opts = \
    {'gradcam': GradCAM,
     'hirescam': HiResCAM,
     'gradcam++' : GradCAMPlusPlus,
     'scorecam': ScoreCAM,
     'xgradcam': XGradCAM,
     'ablationcam': AblationCAM,
     'eigencam': EigenCAM,
     'eigengradcam': EigenGradCAM,
     'layercam': LayerCAM,
     'fullgrad': FullGrad,
     }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='resnet34_CUB_expert')
    parser.add_argument('--data_root', type=str, default='./data/CUB_200_2011/')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--topk',  type=int, default=300)
    parser.add_argument('--copy_from_topk',  type=int, default=None, help='Copy conditional attributions from a larger topk folder')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--methods', type=str, nargs='+', default=None)
    parser.add_argument('--layer_type', type=str, default=None)
    parser.add_argument('--class_subset_file', type=str, default='paper.json')

    # gradcam-based explanation params
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth', action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

    args = parser.parse_args()

    args.pytorch_gradcam_methods = {}
    for method in args.methods:
        if method in pytorch_gradcam_opts:
            args.pytorch_gradcam_methods[method] = pytorch_gradcam_opts[method]

    return args


def main():
    args = parse_args()

    model = load_expert_model(args.exp_name, args.checkpoint_path)
    model.eval()

    with open(os.path.join(f'./class_subsets/{args.class_subset_file}'), 'r') as f:
        class_subset = json.load(f)

    transform = load_dataset_transforms('CUB', eval=True)
    dataset = CUBDataset(root_path=args.data_root, transforms=transform,
                         class_subset=class_subset, return_path=True)

    method_run_bool = False
    if len(args.pytorch_gradcam_methods) > 0:
        run_pytorch_gradcam_methods(args, model, dataset, args.layer_type)
        method_run_bool = True

    if 'crp' in args.methods[0]:
        run_crp(args, model, dataset, class_subset, topk=args.topk, copy_from_topk=args.copy_from_topk)
        method_run_bool = True

    if not method_run_bool:
        raise ValueError(f'{args.methods} not recognized!')


def run_pytorch_gradcam_methods(args, model, dataset, layer_type):
    methods_dict = args.pytorch_gradcam_methods

    model = model.eval()

    # Choose the target layer you want to compute the visualization for.
    layer_type = layer_type
    if layer_type == 'resnet_default':
        target_layers = [model.layer4]
    else:
        target_layers = find_layer_types_recursive(model, [torch.nn.Conv2d])

    folder_name = os.path.abspath(f'explanations/{args.exp_name}/{layer_type}_layers/')
    os.makedirs(folder_name, exist_ok=True)
    for method in methods_dict.keys():
        os.makedirs(os.path.join(folder_name, method), exist_ok=True)

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    for i, method in enumerate(methods_dict.keys()):
        print(f'Running {method}')
        for idx, (img, label, img_path) in enumerate(dataloader):
            print(img_path)
            img = img.to('cuda')
            img_id = img_path[0].split('/')[-1].split('.')[0]

            path = os.path.join(folder_name, method, dataset.class_names[label.item()], img_id)

            os.makedirs(path, exist_ok=True)
            cam_algorithm = methods_dict[method]
            with cam_algorithm(model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda) as cam:
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32
                target_categories = label.cpu().data.numpy()
                targets = [ClassifierOutputTarget(
                    category) for category in target_categories]
                grayscale_cams = cam(input_tensor=img,
                                     targets=targets,
                                     aug_smooth=args.aug_smooth,
                                     eigen_smooth=args.eigen_smooth,
                                     individual_layer_cams=True)
                if method != 'fullgrad':
                    grayscale_cams = np.concatenate(grayscale_cams, axis=1)

                # Here grayscale_cam has only one image in the batch
                for ci, grayscale_cam in enumerate(grayscale_cams[0, :]):
                    if np.max(grayscale_cam) != 0:
                        grayscale_cam = grayscale_cam / np.max(grayscale_cam)
                    grayscale_cam = np.uint8(255 * grayscale_cam)

                    cv2.imwrite(f'{path}/cam_layer{ci}.jpg', grayscale_cam)


def run_crp(args, model, dataset, class_subset, topk, copy_from_topk):
    from interpretability_methods_src.zennit_crp_extended.compute_relevances import compute_relevances, check_relevances
    from interpretability_methods_src.zennit_crp_extended.select_neurons import select_top_k_neurons
    from interpretability_methods_src.zennit_crp_extended.generate_conditional_attributions import \
        save_conditional_attributions_for_dataset
    from interpretability_methods_src.zennit_crp_extended.crp.helper import get_layer_names
    from interpretability_methods_src.zennit_crp_extended.utils import compute_skip_conditions

    from zennit.composites import EpsilonPlusFlat
    from zennit.torchvision import ResNetCanonizer

    layer_type = 'conv'
    folder_name = os.path.abspath(f'explanations/{args.exp_name}/{layer_type}_layers/')
    layer_names, layers = get_layer_names(model, [torch.nn.Conv2d])
    composite = EpsilonPlusFlat([ResNetCanonizer()])
    if check_relevances(folder_name):
        print('Relevances already computed')
    else:
        compute_relevances(model, dataset, composite, layer_names, folder_name)
    skip_conditions = compute_skip_conditions(layer_names, model, dataset, folder_name)
    method = 'average'
    selected_neurons = select_top_k_neurons(method, folder_name, dataset, class_subset, topk=topk)
    save_conditional_attributions_for_dataset(dataset, layer_names, model, composite, selected_neurons, folder_name, topk=topk,
                                              skip_conditions=skip_conditions, num_images_per_class=None, copy_from_topk=copy_from_topk)


if __name__ == '__main__':
    main()
