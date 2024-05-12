from sklearn.decomposition import PCA, NMF
from PIL import Image
import numpy as np
import os
from interpretability_methods_src.zennit_crp.crp import image
from torchnmf.nmf import NMF as TorchNMF
from sklearn.decomposition import NMF
import torch
import time


decomposition_opts = {
    'gpu_nmf': TorchNMF,
    'cpu_nmf': NMF,
}


def make_reduction_folder_name(exp_name, layer_type, reduction_method, num_components, skip_downsample):
    reduction_folder = f'./explanations/{exp_name}/reduction/{layer_type}/'
    _reduction_folder = f'{reduction_method}_ncomp={num_components}'
    if skip_downsample:
        _reduction_folder += '_sdow'
    reduction_folder = os.path.join(reduction_folder, _reduction_folder) + '/'
    return reduction_folder

def reduce(method, n_components, pred_explanations, save_folder, **kwargs):

    explanations_dict = next(iter(pred_explanations.values()))
    img_shape = next(iter(explanations_dict.values())).shape
    h, w = img_shape
    num_exp = len(explanations_dict)

    no_downsample_mask = None
    if kwargs.get('skip_downsample', False):
        # skip downsample layers if specified
        print('skipping downsampling')
        no_downsample_mask = np.stack(['downsample' not in key for key in list(explanations_dict.keys())])
        num_exp = no_downsample_mask.sum()

    if method == 'gpu_nmf':
        red = TorchNMF(Vshape=(h * w, num_exp), rank=n_components)
        red.to('cuda')
    elif method == 'cpu_nmf':
        red = NMF(n_components=n_components)
    else:
        raise NotImplementedError

    for img_id in pred_explanations:
        explanations_dict = pred_explanations[img_id]
        explanation_stack = np.stack(list(explanations_dict.values()))

        if kwargs.get('skip_downsample', False):
            explanation_stack = explanation_stack[no_downsample_mask]

        start_time = time.time()
        flat_img = explanation_stack.reshape(num_exp, h * w)
        print(f'fitting... {img_id}', end='')
        if 'gpu_nmf' == method:
            flat_img = torch.FloatTensor(flat_img).to('cuda').T
            num_iter = red.fit(flat_img, beta=2, max_iter=1000)
        else:
            red.fit(flat_img)

        print(f' {method} took {time.time() - start_time} seconds')
        folder = os.path.join(save_folder, img_id)
        os.makedirs(folder, exist_ok=True)
        for i, comp in enumerate(get_components(red, method)):
            comp = comp.reshape(h, w)
            image.imgify(comp).save(os.path.join(folder, f'component_{i}.png'))

        if method == 'gpu_nmf':
            red.W.data = torch.randn_like(red.W.data).abs()
            red.H.data = torch.randn_like(red.H.data).abs()


def get_components(red, method):

    if 'gpu_nmf' in method:
        return red.H.T.detach().cpu().numpy()
    else:
        return red.components_