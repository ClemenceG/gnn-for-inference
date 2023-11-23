echo "Diagonal marginal distribution visualization"

python run_exps.py --exp_name res_$1 --model_name $2 --train_num $3

# Example: python run_exps.py --exp_name res_ladder_small_ladder_small --model_name mgnn_inference --train_num 1