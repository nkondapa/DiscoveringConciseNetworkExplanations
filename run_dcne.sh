python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 3 --skip_downsample
python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 7 --skip_downsample
python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 10 --skip_downsample

python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 3  --skip_downsample
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 7  --skip_downsample
python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 10  --skip_downsample

python feature_clusters.py --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 10 --skip_downsample

## SWEEP NUM COMPONENTS
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp --reduction_method cpu_nmf --num_components 25 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp --reduction_method cpu_nmf --num_components 50 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp --reduction_method cpu_nmf --num_components 100 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp --reduction_method cpu_nmf --num_components 150 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp --reduction_method cpu_nmf --num_components 200 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp --reduction_method cpu_nmf --num_components 250 --skip_downsample

#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 25 --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 50 --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 100 --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 150 --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 200 --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_300 --reduction_method cpu_nmf --num_components 250 --skip_downsample


## SWEEP BASE SIZE
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_10 --reduction_method cpu_nmf --num_components 10 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_50 --reduction_method cpu_nmf --num_components 10 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_100 --reduction_method cpu_nmf --num_components 10 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_150 --reduction_method cpu_nmf --num_components 10 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_200 --reduction_method cpu_nmf --num_components 10 --skip_downsample
#python reduce_num_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_250 --reduction_method cpu_nmf --num_components 10 --skip_downsample


#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_10 --reduction_method cpu_nmf --num_components 10  --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_50 --reduction_method cpu_nmf --num_components 10  --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_100 --reduction_method cpu_nmf --num_components 10  --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_150 --reduction_method cpu_nmf --num_components 10  --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_200 --reduction_method cpu_nmf --num_components 10  --skip_downsample
#python score_explanations.py --exp_name resnet34_CUB_expert --explanation_method zennit_crp_250 --reduction_method cpu_nmf --num_components 10  --skip_downsample
