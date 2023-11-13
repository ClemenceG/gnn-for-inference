#!/bin/bash -ex
# Edited - from pgm_graph_inference/make_data.sh
# Runner of small exact experiment
# run as run_small.sh make_data path

cd pgm_graph_inference
# rm ./graphical_models/datasets/train/$1 -rf
# rm ./graphical_models/datasets/test/$1 -rf
ARG2=${2:-"py2"}
echo -e "\tRunning with python $ARG2"

if [ $ARG2 = "py3" ]; then
            echo -e "\tRunning with python3"
            echo -e "\tCreating train data"
            python3 create_data.py --graph_struct $1 --size_range 16_16 \
                                --num 5000 --data_mode train --mode marginal --algo exact \
                                --verbose True --base_data_dir ../data/datasets/

            echo -e "\tCreating test data"
            python3 create_data.py --graph_struct $1 --size_range 16_16 \
                                --num 1000 --data_mode test --mode marginal --algo exact \
                                --verbose True --base_data_dir ../data/datasets/
else
            echo -e "\tRunning with python"
            echo -e "\tCreating train data"
            python create_data.py --graph_struct $1 --size_range 16_16 \
                                --num 5000 --data_mode train --mode marginal --algo exact \
                                --verbose True --base_data_dir ../data/datasets/

            echo -e "\tCreating test data"
            python create_data.py --graph_struct $1 --size_range 16_16 \
                                --num 1000 --data_mode test --mode marginal --algo exact \
                                --verbose True --base_data_dir ../data/datasets/
fi
cd ../