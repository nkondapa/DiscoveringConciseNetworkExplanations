python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods gradcam --layer_type conv
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods xgradcam --layer_type conv
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods gradcam++ --layer_type conv
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods layercam --layer_type conv
python generate_explanations.py --exp_name resnet34_CUB_expert --checkpoint_path checkpoints/resnet34_CUB_expert.pth.tar --methods fullgrad --layer_type conv

python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method layercam --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method gradcam --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method gradcam++ --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method xgradcam --layer_type conv_layers
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method fullgrad --layer_type conv_layers