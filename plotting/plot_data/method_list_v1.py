from interpretability_methods_src.reduction import make_reduction_folder_name


method_list = [
    {'method': 'fullgrad', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers'},
    {'method': 'gradcam', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers'},
    {'method': 'gradcam++', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers'},
    {'method': 'layercam', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers', 'rename': 'layercam'},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 'Ours', 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers', 'rename': 'CRP 300'},
    {'method': 'zennit_crp_10', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers', 'rename': 'CRP 10'},
    {'method': 'xgradcam', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers', 'rename': 'xgradcam'},
]


method_list_component_sweep = [
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 3, 'rename': 3, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 7, 'rename': 7, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 10, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 25, 'rename': 25, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 50, 'rename': 50, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 100, 'rename': 100, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 150, 'rename': 150, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 200, 'rename': 200, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 250, 'rename': 250, 'skip_downsample': True},

]


method_list_base_sweep = [
    {'method': 'zennit_crp_10', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 10, 'skip_downsample': True},
    {'method': 'zennit_crp_50', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 50, 'skip_downsample': True},
    {'method': 'zennit_crp_100', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 100, 'skip_downsample': True},
    {'method': 'zennit_crp_150', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 150, 'skip_downsample': True},
    {'method': 'zennit_crp_200', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 200, 'skip_downsample': True},
    {'method': 'zennit_crp_250', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 250, 'skip_downsample': True},
    {'method': 'zennit_crp_300', 'exp_name': 'resnet34_CUB_expert', 'layer_type': 'conv_layers',
     'reduction_method': 'cpu_nmf', 'num_components': 10, 'rename': 300, 'skip_downsample': True},
]


def construct_path(exp_name, layer_type, reduction_method=None, num_components=None, skip_downsample=False):
    if reduction_method is None:
        return f'./explanations/{exp_name}/{layer_type}/'
    else:
        return make_reduction_folder_name(exp_name, layer_type, reduction_method, num_components, skip_downsample)