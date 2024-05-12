python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods crp_300 --topk 300
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods crp_10 --topk 10 --copy_from_topk 300
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods crp_50 --topk 50 --copy_from_topk 300
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods crp_100 --topk 100 --copy_from_topk 300
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods crp_150 --topk 150 --copy_from_topk 300
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods crp_200 --topk 200 --copy_from_topk 300
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods crp_250 --topk 250 --copy_from_topk 300


python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_10 --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_50 --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_100 --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_150 --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_200 --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_250 --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --layer_type conv_layers

