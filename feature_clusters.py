import argparse

from datasets.cub import CUBDataset

from interpretability_methods_src.utils import load_explanations, select_threshold
from interpretability_methods_src.reduction import make_reduction_folder_name
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from interpretability_methods_src.zennit_crp.crp.image import gaussian_blur, max_norm, zimage

from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
import matplotlib.pyplot as plt
import os
import pickle as pkl
from plotting import plot_vars
from sklearn.decomposition import PCA, KernelPCA
import torchvision
from PIL import Image, ImageDraw
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='resnet34_CUB_expert')
    parser.add_argument('--data_root', type=str, default='./data/CUB_200_2011/')

    parser.add_argument('--layer_type', type=str, default='conv_layers')
    parser.add_argument('--explanation_method', type=str, default='zennit_crp_300')
    parser.add_argument('--keep_all', action='store_true', default=False)
    parser.add_argument('--entangled', action='store_true', default=False)

    parser.add_argument('--reduction_method', type=str, default=None)
    parser.add_argument('--num_components', type=int, default=None)
    parser.add_argument('--skip_downsample', action='store_true', default=False)
    parser.add_argument('--class_subset_file', type=str, default='paper.json')

    # ground truth annotation params
    parser.add_argument('--annotator_num', type=int, default=1)
    parser.add_argument('--combine_gt', action='store_true', default=False)
    parser.add_argument('--max_r_combine', type=int, default=3)
    parser.add_argument('--resize', type=int, nargs=2, default=(448, 448))
    parser.add_argument('--smooth', action='store_true', default=False)

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
    dataset = CUBDataset(root_path=args.data_root, transforms=None,
                         class_subset=class_subset, return_path=True)

    for class_index in class_subset:
        class_name = dataset.classes[class_index].split('.')[-1]
        explanation_main_folder = f'./explanations/{args.exp_name}/{args.layer_type}/'

        explanation_folder = explanation_main_folder + f'{args.explanation_method}/' + class_name
        reduction_main_folder = make_reduction_folder_name(args.exp_name, args.layer_type, args.reduction_method,
                                                           args.num_components, args.skip_downsample)
        reduction_folder = reduction_main_folder + f'{args.explanation_method}/' + class_name
        image_folder = os.path.join(dataset.root_path, 'images', dataset.classes[class_index])

        cluster_summaries_folder = reduction_folder.replace('./explanations', './cluster_summaries')
        os.makedirs(cluster_summaries_folder, exist_ok=True)
        clusters_folder = os.path.join(cluster_summaries_folder, 'clusters')
        figures_folder = os.path.join(cluster_summaries_folder, 'figures')
        data_folder = os.path.join(cluster_summaries_folder, 'data')
        os.makedirs(clusters_folder, exist_ok=True)
        os.makedirs(figures_folder, exist_ok=True)
        os.makedirs(data_folder, exist_ok=True)

        try:
            with open(os.path.join(data_folder, 'data_matrix.pkl'), 'rb') as f:
                R, feature_ids, neuron_ids = pkl.load(f)
        except FileNotFoundError:
            explanations = load_explanations(explanation_folder)
            reduction_explanations = load_explanations(reduction_folder)
            R, feature_ids, neuron_ids = compute_sparse_matrix(reduction_explanations, explanations)
            with open(os.path.join(data_folder, 'data_matrix.pkl'), 'wb') as f:
                pkl.dump([R, feature_ids, neuron_ids], f)

        data = R.todense()
        cluster_params = {'eps': 1.4, 'min_samples': 5}
        cluster_fn = f'clusters_{cluster_params["eps"]}_{cluster_params["min_samples"]}.pkl'

        try:
            # raise FileNotFoundError
            with open(os.path.join(clusters_folder, cluster_fn), 'rb') as f:
                clusters = pkl.load(f)
        except FileNotFoundError:
            clusters = compute_clusters(data, cluster_params)
            with open(os.path.join(clusters_folder, cluster_fn), 'wb') as f:
                pkl.dump(clusters, f)

        annotator_num = args.annotator_num
        combine_gt = args.combine_gt
        max_r_combine = args.max_r_combine
        smooth = args.smooth
        resize = args.resize

        s = make_experiment_score_str(class_index, annotator_num, combine_gt, max_r_combine, smooth)
        score_folder = os.path.join(reduction_main_folder, 'scores', args.explanation_method)
        score_name = f'scores{s}.pkl'
        best_match_name = f'best_match{s}.pkl'
        with open(os.path.join(score_folder, score_name), 'rb') as f:
            scores = pkl.load(f)

        tmp = select_threshold(scores)
        threshold = tmp['threshold']
        thresholded_scores = {}
        for img_id in scores:
            thresholded_scores[img_id] = scores[img_id][threshold]

        visualize_clusters(data, clusters, feature_ids, thresholded_scores, class_index, cluster_params, figures_folder, ignore_outliers=True, show=False)
        visualize_clustered_images(image_folder, reduction_folder, clusters, feature_ids, class_name, cluster_params, figures_folder,
                                       max_num_samples=5, show=False)


def compute_sparse_matrix(concise_explanations, base_explanations):

    neuron_ids = set()
    for img_id in base_explanations:
        for neuron_id in base_explanations[img_id]:
            neuron_ids.add(str(neuron_id))

    neuron_ids = np.array(list(neuron_ids))

    feature_ids = set()
    for img_id in concise_explanations:
        for feature_id in concise_explanations[img_id]:
            feature_ids.add(str((img_id, feature_id)))

    feature_ids = np.array(list(feature_ids))

    row_indices = []
    col_indices = []
    data = []
    for img_id in concise_explanations:
        print(img_id)
        dm = np.array(list(concise_explanations[img_id].values()))
        sm = np.array(list(base_explanations[img_id].values()))

        dm.reshape(dm.shape[0], -1)
        sm.reshape(sm.shape[0], -1)
        cd = cosine_distances(dm.reshape(dm.shape[0], -1), sm.reshape(sm.shape[0], -1))

        for si, sm_key in enumerate(base_explanations[img_id]):
            for di, dm_key in enumerate(concise_explanations[img_id]):
                sm_ind = np.where(neuron_ids == str(sm_key))[0][0]
                dm_ind = np.where(feature_ids == str((img_id, dm_key)))[0][0]
                # matrix[dm_ind, sm_ind] = 1 - cd[di, si]
                row_indices.append(dm_ind)
                col_indices.append(sm_ind)
                data.append(1 - cd[di, si])

    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)
    data = np.array(data)

    R = coo_matrix((data, (row_indices, col_indices)), shape=(len(feature_ids), len(neuron_ids)))
    return R, feature_ids, neuron_ids


def compute_clusters(data, cluster_params):


    eps = cluster_params.get('eps')
    min_samples = cluster_params.get('min_samples')

    # d = k_distances(data, n=min_samples)
    # plt.plot(d)
    # plt.ylabel("k-distances")
    # plt.grid(True)
    # plt.savefig()
    clusters = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit_predict(np.asarray(data))

    return clusters


def within_cluster_iou(clusters, feature_ids, scores, feature_list=None, ignore_outliers=True):
    # scores, score_name = self.load_scores()
    cluster_scores = {}

    for p_clusters_i in np.unique(clusters):
        if p_clusters_i == -1 and ignore_outliers:
            continue
        cluster_scores[p_clusters_i] = {}

    for fi, feature_id in enumerate(feature_ids):
        img_id, component = eval(feature_id)
        p_clusters_i = clusters[fi]
        if p_clusters_i == -1 and ignore_outliers:
            continue
        cluster_scores[p_clusters_i] = cluster_scores.get(p_clusters_i, {})
        for segmentation_key in scores[img_id][component]:
            if feature_list is not None and segmentation_key not in feature_list:
                continue
            cluster_scores[p_clusters_i][segmentation_key] = cluster_scores[p_clusters_i].get(segmentation_key, [])
            cluster_scores[p_clusters_i][segmentation_key].append(scores[img_id][component][segmentation_key])

    print(cluster_scores.keys())
    arr_mean = np.zeros(shape=(len(cluster_scores.keys()), len(feature_list)))
    arr_std = np.zeros(shape=(len(cluster_scores.keys()), len(feature_list)))
    for pi, p_clusters_i in enumerate(cluster_scores.keys()):
        for si, segmentation_key in enumerate(cluster_scores[p_clusters_i]):
            print(pi, si, cluster_scores[p_clusters_i].keys())
            cluster_scores[p_clusters_i][segmentation_key] = np.array(cluster_scores[p_clusters_i][segmentation_key])
            arr_mean[pi, si] = cluster_scores[p_clusters_i][segmentation_key].mean()
            arr_std[pi, si] = cluster_scores[p_clusters_i][segmentation_key].std()
            print(p_clusters_i, segmentation_key, cluster_scores[p_clusters_i][segmentation_key].mean(), cluster_scores[p_clusters_i][segmentation_key].std())

    return arr_mean, arr_std, cluster_scores


def visualize_clusters(data, clusters, feature_ids, scores, target_class, cluster_params, figures_folder, ignore_outliers=True, show=False):

    feature_list = plot_vars.brushes[target_class]

    arr_mean, arr_std, cluster_scores = within_cluster_iou(clusters, feature_ids, scores, feature_list, ignore_outliers=ignore_outliers)
    first_key = list(cluster_scores.keys())[0]
    eps = cluster_params.get('eps')
    min_samples = cluster_params.get('min_samples')



    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    p = axes.imshow(arr_mean, cmap='magma')
    axes.set_xticks(np.arange(len(cluster_scores[first_key].keys())))
    axes.set_xticklabels(list(map(lambda x: x.lower(), cluster_scores[first_key].keys())), rotation=30, fontsize=12, ha='right')
    axes.set_yticks(np.arange(arr_mean.shape[0]))

    axes.set_ylabel('Cluster', fontsize=12)
    plt.colorbar(p, label='Mean IoU')
    plt.tight_layout()
    if not show:
        plt.savefig(os.path.join(figures_folder, f'cluster_scores_{eps}_{min_samples}.pdf'), dpi=300)
        plt.close()

    data = np.asarray(data)
    p_pca_embed = PCA(n_components=2).fit_transform(data)
    p_rbf_embed = KernelPCA(n_components=2, kernel='rbf').fit_transform(data)
    fig, axes = plt.subplots(1, 2)
    for p_clusters_i in np.unique(clusters):
        mask = clusters == p_clusters_i
        axes[0].scatter(p_pca_embed[mask, 0], p_pca_embed[mask, 1], label=p_clusters_i)
        axes[1].scatter(p_rbf_embed[mask, 0], p_rbf_embed[mask, 1])

    axes[0].set_title('PCA')
    axes[1].set_title('RBF - PCA')
    axes[0].legend()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(figures_folder, f'cluster_vis_{eps}_{min_samples}.pdf'), dpi=300)


def visualize_clustered_images(img_folder, reduction_folder, clusters, feature_ids, class_name, cluster_params, figures_folder, max_num_samples=5, show=False):

    eps = cluster_params.get('eps')
    min_samples = cluster_params.get('min_samples')


    for p_clusters_i in np.unique(clusters):
        mask = clusters == p_clusters_i
        ims = np.array(list(map(eval, feature_ids[mask])))
        print(len(ims))
        num_ims = min(max_num_samples, len(ims))
        fig, axes = plt.subplots(1, num_ims, squeeze=False)
        fig.set_size_inches(15, 5)
        axes[0].flatten()[0].set_xlabel(class_name)
        # axes = axes

        for idx, im in enumerate(ims):
            if idx >= num_ims:
                break

            img_id, component = im
            img_pil = Image.open(os.path.join(img_folder, img_id + '.jpg')).resize((448, 448))
            heatmap = Image.open(os.path.join(reduction_folder, img_id, f'{component}.png'))
            img_tensor = torchvision.transforms.ToTensor()(img_pil)
            heatmap = torchvision.transforms.ToTensor()(heatmap)

            explanation_overlayed = apply_heatmap(img_tensor, [heatmap], alpha=0.3, vis_th=0.3, kernel_size=19)[0]
            axes[0].flatten()[idx].imshow(explanation_overlayed)

            axes[0].flatten()[idx].set_axis_off()
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(os.path.join(figures_folder, f'cluster_sample_{p_clusters_i}_{eps}_{min_samples}.pdf'),
                        dpi=600)


def max_norm(rel, stabilize=1e-10):
    return rel / (rel.max() + stabilize)


def apply_heatmap(image, heatmaps, alpha=0.3, vis_th=0.2, kernel_size=19):
    imgs = []
    for i in range(len(heatmaps)):
        img = image.clone()
        filtered_heat = max_norm(gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0])
        vis_mask = filtered_heat > vis_th

        inv_mask = ~vis_mask
        img = img * vis_mask + img * inv_mask * alpha
        img = zimage.imgify(img.detach().cpu())
        imgs.append(img)
    return imgs


if __name__ == '__main__':
    main()