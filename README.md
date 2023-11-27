# Reproduction of "Inference in Probabilistic Graphical Models by Graph Neural Networks"
*Collaborators*: Anita Kriz, Alireza Dizaji, Anthony Gosselin, Jeremy Qin, Clémence Granade

This repository contains code and resources to reproduce the results presented in the research paper "Inference in Probabilistic Graphical Models by Graph Neural Networks" by KiJung Yoon, et al. in 2009.

## Getting Started
1. Clone this repository: `git clone https://github.com/ClemenceG/gnn-for-inference.git`
2. Install requirements from requirements.txt
```bash
pip install -r requirements.txt
```
4. Generate data (option py3 for python3)
```bash
./make_data.sh grid [py3]
```
5. Before running experiments, make sure to have a folder named `experiments/saved_exp_res/`. Results will be logged as npy file in this folder.
6. To run experiments (ex: grid-9): `$./run_all.sh grid_small MODEL TRAIN_NUM`
7. To view results for a specific npy file, run the following: `python3 read_results.py --file_path LOGS_PATH`

### Commands
- `./make_data.sh`: generates data (edited from sunfanyunn's repo)

## Paper Information

- **Title:** Inference in Probabilistic Graphical Models by Graph Neural Networks
- **Authors:** KiJung Yoon, Renjie Liao, Yuwen Xiong, Lisa Zhang, Ethan Fetaya, Raquel Urtasun, Richard Zemel, Xaq Pitkow
- **Year:** 2019

- **[Paper Link](https://arxiv.org/pdf/1803.07710.pdf)**
- **[Code Repository](https://github.com/fanyun-sun/pgm_graph_inference)**

The paper discusses the utilization of Graph Neural Networks for inference in probabilistic graphical models. This repository aims to reproduce the experiments and results detailed in the paper.



## Contents

The repository is organized as follows:

- **`code/`**: Contains code for replicating the experiments and running the models.
- **`data/`**: Placeholder for any necessary data or instructions to obtain the data used in the experiments.
- **`results/`**: Holds generated results, outputs, and comparisons.


