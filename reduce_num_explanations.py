import argparse
import os
from datasets.cub import CUBDataset
from interpretability_methods_src.pytorch_grad_cam import (
    GradCAM, HiResCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, AblationCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad
)
from interpretability_methods_src.reduction import reduce, make_reduction_folder_name, decomposition_opts
from interpretability_methods_src.utils import load_explanations
import json

pytorch_gradcam_opts = \
    {'gradcam': GradCAM,
     'hirescam': HiResCAM,
     'gradcam++': GradCAMPlusPlus,
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

    parser.add_argument('--layer_type', type=str, default='conv_layers')
    parser.add_argument('--explanation_method', type=str, default='zennit_crp_300')
    parser.add_argument('--reduction_method', type=str, default='gpu_nmf')
    parser.add_argument('--num_components', type=int, default=3)
    parser.add_argument('--skip_downsample', action='store_true', default=False)
    parser.add_argument('--class_subset_file', type=str, default='paper.json')

    args = parser.parse_args()

    return args




def main():
    args = parse_args()

    with open(os.path.join(f'./class_subsets/{args.class_subset_file}'), 'r') as f:
        class_subset = json.load(f)

    dataset = CUBDataset(root_path=args.data_root, transforms=None,
                         class_subset=class_subset, return_path=True)

    explanation_main_folder = f'./explanations/{args.exp_name}/{args.layer_type}/'
    reduction_main_folder = make_reduction_folder_name(args.exp_name, args.layer_type, args.reduction_method,
                                                  args.num_components, args.skip_downsample)

    for class_index in class_subset:
        explanation_folder = os.path.join(explanation_main_folder + f'{args.explanation_method}/',
                                          dataset.classes[class_index].split('.')[-1])
        pred_explanations = load_explanations(explanation_folder)

        save_folder = os.path.join(reduction_main_folder + f'{args.explanation_method}/',
                                          dataset.classes[class_index].split('.')[-1])
        reduce(args.reduction_method, args.num_components, pred_explanations, save_folder, skip_downsample=args.skip_downsample)


if __name__ == '__main__':
    main()
