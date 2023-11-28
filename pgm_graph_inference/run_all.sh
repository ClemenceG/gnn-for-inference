#!/bin/bash -ex
echo -e "\tTraining your GNN"
python train.py --train_set_name $1 --mode marginal --epochs 50 --verbose True --model_name $2 --train_num $3

echo -e "\tRunning tests"
python run_exps.py --exp_name in_sample_$1 --model_name $2 --train_num $3 --test_num $4
