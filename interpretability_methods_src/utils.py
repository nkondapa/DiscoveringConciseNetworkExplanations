import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def load_explanations(folder):
    img_folders = sorted(os.listdir(os.path.join(folder)))
    explanation_maps = {}
    pbar = tqdm(total=len(img_folders), dynamic_ncols=True)
    for img_folder in img_folders:
        pbar.update(1)
        explanation_map_path = os.path.join(folder, img_folder)
        explanation_maps[img_folder] = {}
        for p in os.listdir(os.path.join(explanation_map_path)):
            heatmap = np.array(Image.open(os.path.join(explanation_map_path, p)))
            explanation_maps[img_folder]['.'.join(p.split('.')[:-1])] = heatmap

    return explanation_maps

def convert_scores_to_matrix(scores, threshold):
    '''

    :param scores: scores should be a dict[img_id][component_i][brush_label] = iou
    :return:
    '''

    # bit of code to pre-allocate the score matrix, need to get maximum num_components,
    # maximum number of brushes, and remove any images that don't have an entry
    num_explanations = 0
    brush_set = set()
    new_scores = {}
    for img_id in sorted(scores.keys()):
        _scores = scores[img_id][threshold]
        if _scores is not None:
            new_scores[img_id] = _scores
            num_explanations = max(num_explanations, len(_scores.keys()))
        else:
            continue

        for ci, emap_i in enumerate(sorted(_scores.keys())):
            brush_set.update(_scores[emap_i].keys())

    scores = new_scores
    brush_map = dict(zip(sorted(brush_set), range(len(brush_set))))
    num_imgs = len(scores.keys())

    score_matrix = np.zeros((num_imgs, num_explanations, len(brush_map)))
    score_matrix[:] = np.nan
    for ii, img_id in enumerate(sorted(scores.keys())):
        _scores = scores[img_id]
        for ci, emap_i in enumerate(sorted(_scores.keys())):
            for i, brush_label in enumerate(sorted(_scores[emap_i].keys())):
                brush_idx = brush_map[brush_label]
                score_matrix[ii, ci, brush_idx] = scores[img_id][emap_i][brush_label]

    return score_matrix, brush_map

def compute_method_comparison_matrix(score_matrix_dict, brush_map):
    """
    Reduce the score matrix to a vector
    score matrix is a matrix of img x component x brush_label
    we take the max over component, then average over img
    :param score_matrix_dict: dict of score_matrix, key is the method name, values is score_matrix
    :param brush_map:
    :return:
    """

    method_names = list(sorted(score_matrix_dict.keys()))
    num_methods = len(method_names)
    num_brush_labels = score_matrix_dict[method_names[0]].shape[-1]

    method_comparison_matrix = np.zeros((num_methods, num_brush_labels))
    cost = []
    for mi, method_name in enumerate(method_names):
        method_comparison_matrix[mi, :] = np.nanmean(np.nanmax(score_matrix_dict[method_name], axis=1), axis=0)
        cost.append(score_matrix_dict[method_name].shape[1])

    # add average to the columns
    method_comparison_matrix = np.hstack(
        (method_comparison_matrix, np.nanmean(method_comparison_matrix, axis=1)[:, np.newaxis]))
    brushes = list(sorted(brush_map.keys()))
    brushes.append('average')

    return method_comparison_matrix, method_names, brushes, cost


def select_threshold(score_dict):
    img_id = list(score_dict.keys())[0]
    threshold_list = list(score_dict[img_id].keys())

    score_threshold_dict = {}
    for threshold in sorted(threshold_list):
        score_matrix, brush_map = convert_scores_to_matrix(score_dict, threshold)
        score_threshold_dict[threshold] = score_matrix

    # best_thresholds = {}
    holdin_score_dict = {}
    inds = np.arange(score_threshold_dict[threshold_list[0]].shape[0])
    hold_in = inds[:int(0.33 * len(inds))]
    hold_out = inds[int(0.33 * len(inds)):]
    for thresh in score_threshold_dict:
        num_ims = score_threshold_dict[thresh].shape[0]
        holdin_score_dict[thresh] = score_threshold_dict[thresh][:]

    threshold_comparison_matrix, thresholds, features, cost = compute_method_comparison_matrix(holdin_score_dict, brush_map)
    threshold_idx = np.argmax(threshold_comparison_matrix[:, -1])
    threshold = threshold_list[threshold_idx]
    return dict(threshold=threshold, score_matrix=score_threshold_dict[threshold], hold_out_indices=hold_out, features=brush_map)
