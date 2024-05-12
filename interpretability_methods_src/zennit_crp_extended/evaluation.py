import os
import json
import glob
from interpretability_methods_src.zennit_crp_extended import rle
import numpy as np
import cv2
from itertools import combinations


def get_explanation_map_folder(explanation_map_folder, dataset, class_idx, annotator_num):
    '''
    Get explanation map folder for a given class
    :param class_idx: the index of the class
    :return:
    '''

    class_name = dataset.classes[class_idx]
    annotator_str = f'annotator{annotator_num}'
    class_explanation_map_folder = os.path.join(explanation_map_folder, annotator_str, class_name)
    return class_explanation_map_folder


def load_ground_truth_explanation_maps(explanation_map_folder, dataset, class_idx, annotator_num=1, combine=True,
                                    max_r_combine=3, overwrite=False):
    '''
    Load ground truth explanation maps for a given class
    :param class_idx: class index
    :param annotator_num: annotator number (only 1 or 2)
    :param combine: should multiple features be combined to test combinations of features?
    :param max_r_combine: is there a maximum number of combinations we should test?
    :param overwrite: overwrite previously generated refactored annotations file
    :return: ground truth explanation maps
    '''

    # get the folder where the explanation maps are stored
    class_explanation_map_folder = get_explanation_map_folder(explanation_map_folder, dataset, class_idx, annotator_num)
    try:
        if overwrite:
            raise FileNotFoundError

        with open(os.path.join(class_explanation_map_folder, 'annotations.json'), 'r') as f:
            ground_truth_explanation_maps = json.load(f)

    except FileNotFoundError:
        print('No organized annotations dict. Refactoring...')
        ground_truth_explanation_maps = refactor_annotations(class_explanation_map_folder)

    # convert all RLE to mask, returns a dictionary of maps indexed by [img_id][gt_explanation_map_key (feature)]
    convert_gtsm_rle_to_mask(ground_truth_explanation_maps, combine, max_r_combine)
    return ground_truth_explanation_maps


def convert_gtsm_rle_to_mask(ground_truth_explanation_maps, combine=True, max_r_combine=None):
    for img_id, img_dict in ground_truth_explanation_maps.items():
        for gt_explanation_map_key, gt_explanation_map_list in img_dict.items():
            if gt_explanation_map_key == 'shape':
                continue
            mask = None
            for gt_explanation_map in gt_explanation_map_list:
                if mask is None:
                    mask = rle.decode_rle(gt_explanation_map)
                else:
                    mask += rle.decode_rle(gt_explanation_map)
            mask[mask > 0] = 1
            mask = mask.reshape(*img_dict['shape'], 4)[:, :, 0]
            # plt.title(img_id)
            # plt.imshow(mask)
            # plt.show()
            ground_truth_explanation_maps[img_id][gt_explanation_map_key] = mask

        if combine:
            combine_gtsms(ground_truth_explanation_maps, img_id, max_r_combine=max_r_combine)


def combine_gtsms(ground_truth_explanation_maps, img_id, max_r_combine=None):
    '''
    Combine all ground truth explanation maps for a given image
    :param ground_truth_explanation_maps:
    :param img_id:
    :return:
    '''

    keys = sorted(list(ground_truth_explanation_maps[img_id].keys()))
    if 'shape' in keys:
        keys.remove('shape')

    max_r_combine = (min(max_r_combine, len(keys)) + 1) if max_r_combine is not None else (len(keys) + 1)

    combs = []
    for i in range(2, max_r_combine):
        combs.extend(combinations(keys, i))

    for comb in combs:
        ground_truth_explanation_maps[img_id]['_'.join(comb)] = None
        for comb_item in comb:
            if ground_truth_explanation_maps[img_id]['_'.join(comb)] is None:
                ground_truth_explanation_maps[img_id]['_'.join(comb)] = ground_truth_explanation_maps[img_id][
                    comb_item].copy()
            else:
                ground_truth_explanation_maps[img_id]['_'.join(comb)] += ground_truth_explanation_maps[img_id][comb_item]
        ground_truth_explanation_maps[img_id]['_'.join(comb)] = (
                ground_truth_explanation_maps[img_id]['_'.join(comb)] > 0).astype(np.uint8)

        # plt.title('_'.join(comb))
        # plt.imshow(ground_truth_explanation_maps[img_id]['_'.join(comb)])
        # plt.show()


def compute_iou(explanation_map, ground_truth_explanation_map, thresholds, smooth):
    """
    Computes the IOU of a explanation map and a ground truth explanation map
    :param explanation_map: The explanation map
    :param ground_truth_explanation_map: The ground truth explanation map
    :return: The IOU
    """

    # convert explanation map to uint8 and scale to 255
    if explanation_map.dtype != np.uint8:
        assert explanation_map.max() <= 1.01, 'explanation map values should be in [0, 1] if not uint8'
        explanation_map = (explanation_map * 255).astype(np.uint8)

    # compute intersection over union metric
    sm_reshaped = cv2.resize(explanation_map, ground_truth_explanation_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
    if smooth:
        cv2.boxFilter(sm_reshaped, 0, (80, 80), sm_reshaped, (-1, -1), False, cv2.BORDER_DEFAULT)

    iou_score_dict = {}
    for threshold in thresholds:
        sm = (sm_reshaped > threshold).astype(np.uint8)
        intersection = np.logical_and(sm, ground_truth_explanation_map)
        union = np.logical_or(sm, ground_truth_explanation_map)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_score_dict[threshold] = iou_score

    return iou_score_dict


def explanation_map_evaluation(ground_truth_explanation_maps, img_id, predicted_explanation_maps, combine_gt=True, max_r_combine=3,
                            thresholds=None, smooth=False):

    if img_id not in ground_truth_explanation_maps:
        print('Image {} not in ground truth explanation maps'.format(img_id))
        return None
    print('Computing IOU for image {}'.format(img_id))

    # compute mean IOU for each explanation map
    '''
    iou_dict structure is as follows:
    [threshold][pred_explanation_map_name][gt_map_name (brushlabel)] = (iou)
    '''

    mean_iou = {}
    best_match = {}
    for threshold in thresholds:
        mean_iou[threshold] = {}
        best_match[threshold] = {}

    import time
    for explanation_map_key, explanation_map in predicted_explanation_maps.items():
        for gt_explanation_map_key, gt_explanation_map in ground_truth_explanation_maps[img_id].items():
            if gt_explanation_map_key == 'shape':
                continue

            # st = time.time()
            # print('Computing score ' )
            _scores = compute_iou(explanation_map, gt_explanation_map, thresholds, smooth)
            # print('Computed...', time.time() - st)
            for threshold in thresholds:
                if explanation_map_key not in mean_iou[threshold]:
                    mean_iou[threshold][explanation_map_key] = {}

                mean_iou[threshold][explanation_map_key][gt_explanation_map_key] = _scores[threshold]

                if gt_explanation_map_key not in best_match[threshold]:
                    best_match[threshold][gt_explanation_map_key] = [explanation_map_key, _scores[threshold]]
                else:
                    if _scores[threshold] > best_match[threshold][gt_explanation_map_key][1]:
                        best_match[threshold][gt_explanation_map_key] = [explanation_map_key, _scores[threshold]]

    return mean_iou, best_match


def refactor_annotations(class_explanation_map_folder):
    files = glob.glob(os.path.join(class_explanation_map_folder, '*.json'))
    with open(files[0], 'r') as f:
        tmp = json.load(f)

    annotation_dict = {}
    for annotation in tmp:
        img_filename = annotation['image'].split('/')[-1].split('-')[-1].split('.')[0]
        annotation_dict[img_filename] = annotation_dict.get(img_filename, {})
        for tag in annotation['tag']:
            assert len(tag['brushlabels']) == 1
            fmt = tag['format']
            # annotation_dict[img_filename][tag['brushlabels'][0]] = tag[fmt]
            annotation_dict[img_filename][tag['brushlabels'][0]] = annotation_dict[img_filename].get(
                tag['brushlabels'][0], [])
            annotation_dict[img_filename][tag['brushlabels'][0]].append(tag[fmt])
            if 'shape' not in annotation_dict[img_filename]:
                annotation_dict[img_filename]['shape'] = (tag['original_height'], tag['original_width'])

    with open(os.path.join(class_explanation_map_folder, 'annotations.json'), 'w') as f:
        json.dump(annotation_dict, f)

    return annotation_dict


def refactor_score_dict(self, score_dict, thresholds):
    '''
    Refactor the score dict to be a dict[img_id][component_i][brush_label] = iou
    :return:
    '''

    new_score_dict = {}
    for img_id in score_dict.keys():
        for threshold in score_dict[img_id]:
            if threshold not in new_score_dict:
                new_score_dict[threshold] = {}
            new_score_dict[threshold][img_id] = score_dict[img_id][threshold]

    return new_score_dict
