import argparse
import os
import numpy as np
from datasets.cub import CUBDataset
from interpretability_methods_src.pytorch_grad_cam import (
    GradCAM, HiResCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, AblationCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad
)
from interpretability_methods_src.zennit_crp_extended import evaluation
import time
import pickle as pkl
from interpretability_methods_src.utils import load_explanations
from interpretability_methods_src.reduction import make_reduction_folder_name
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

    parser.add_argument('--reduction_method', type=str, default=None)
    parser.add_argument('--num_components', type=int, default=None)
    parser.add_argument('--skip_downsample', action='store_true', default=False)
    parser.add_argument('--class_subset_file', type=str, default='paper.json')

    args = parser.parse_args()

    return args


def make_experiment_score_str(class_idx, annotator_num, combine_gt=False, max_r_combine=3, smooth=False):
    s = f''
    s += f'_class={class_idx}'
    s += f'_an={annotator_num}'
    if combine_gt:
        s += f'_combine={max_r_combine}'
    if smooth:
        s += '_smooth'

    return s

def main():
    args = parse_args()

    with open(os.path.join(f'./class_subsets/{args.class_subset_file}'), 'r') as f:
        class_subset = json.load(f)

    map_path = os.path.join(args.data_root, 'saliency_maps')
    dataset = CUBDataset(root_path=args.data_root, transforms=None,
                         class_subset=class_subset, return_path=True)
    map_thresholds = np.array([0, 25, 50, 100, 150, 200, 250])
    annotator_num = 1
    combine_gt = False
    max_r_combine = 3
    smooth = False

    explanation_main_folder = f'./explanations/{args.exp_name}/{args.layer_type}/'
    if args.reduction_method is not None:
        reduction_main_folder = make_reduction_folder_name(args.exp_name, args.layer_type, args.reduction_method,
                                                              args.num_components, args.skip_downsample)
        explanation_main_folder = reduction_main_folder

    for class_index in class_subset:
        explanation_folder = os.path.join(explanation_main_folder + f'{args.explanation_method}/',
                                          dataset.classes[class_index].split('.')[-1])

        gt_explanations, pred_explanations, scores, best_matches = evaluate(map_path, explanation_folder, dataset,
                                                                            class_index,
                                                                            annotator_num=annotator_num,
                                                                            thresholds=map_thresholds,
                                                                            combine_gt=combine_gt,
                                                                            max_r_combine=max_r_combine,
                                                                            smooth=smooth)

        # save
        folder = os.path.join(explanation_main_folder, 'scores', args.explanation_method)
        os.makedirs(folder, exist_ok=True)

        s = make_experiment_score_str(class_index, annotator_num, combine_gt, max_r_combine, smooth)
        score_name = f'scores{s}.pkl'
        best_match_name = f'best_match{s}.pkl'
        with open(os.path.join(folder, score_name), 'wb') as f:
            pkl.dump(scores, f)
        with open(os.path.join(folder, best_match_name), 'wb') as f:
            pkl.dump(best_matches, f)


def evaluate(saliency_map_folder, explanations_folder, dataset, class_idx, annotator_num, thresholds=None,
             combine_gt=False, max_r_combine=3, smooth=False):
    gt_explanations = evaluation.load_ground_truth_explanation_maps(saliency_map_folder, dataset, class_idx,
                                                                    annotator_num=annotator_num, combine=combine_gt,
                                                                    max_r_combine=max_r_combine)

    '''
    scores structure is as follows:
    [img_filename][threshold][pred_explanation_map_name][gt_map_name] = iou 
    ie: scores['Black_Footed_Albatross_0090_796077'][0]['layer4.2.conv2_269']['dark body'] = 0.5
    
    best matches structure is as follows:
    [img_filename][threshold][gt_map_name] = [pred_explanation_map_name, iou]
    ie: best_matches['Black_Footed_Albatross_0090_796077'][0]['dark body'] = ['layer4.2.conv2_269', 0.5]
    '''

    explanations = load_explanations(explanations_folder)

    scores = {}
    best_matches = {}
    start_time = time.time()
    for idx, (img_id, img_pred_sm) in enumerate(explanations.items()):
        print(f'{idx}/{len(explanations)} [{time.time() - start_time}] ', end='')
        score, best_match = evaluation.explanation_map_evaluation(gt_explanations, img_id, img_pred_sm,
                                                                  thresholds=thresholds, smooth=smooth)
        if score is not None:
            scores[img_id] = score
            best_matches[img_id] = best_match

    return gt_explanations, explanations, scores, best_matches


if __name__ == '__main__':
    main()
