# Reproduction of "Inference in Probabilistic Graphical Models by Graph Neural Networks"

This repository contains code and resources to reproduce the results presented in the research paper "Inference in Probabilistic Graphical Models by Graph Neural Networks" by KiJung Yoon, et al. in 2009.

## Getting Started
1. Clone this repository: `git clone https://github.com/ClemenceG/gnn-for-inference.git`
2. Run `./pull_sunfanyunn_repo.sh`
3. Install requirements from requirements.txt
```bash
pip install -r requirements.txt
```
4. Generate data (option py3 for python3)
```bash
./make_data.sh grid [py3]
```

### Commands
- `./pull_sunfanyunn_repo.sh`: pulls code from https://github.com/fanyun-sun/pgm_graph_inference.git to obtain data
- `./make_data.sh`: generates data (edited from sunfanyunn's repo)

## Paper Information

- **Title:** Inference in Probabilistic Graphical Models by Graph Neural Networks
- **Authors:** KiJung Yoon, Renjie Liao, Yuwen Xiong, Lisa Zhang, Ethan Fetaya, Raquel Urtasun, Richard Zemel, Xaq Pitkow
- **Year:** 2009

- **[Paper Link](https://arxiv.org/pdf/1803.07710.pdf)**
- **[Code Repository](https://github.com/fanyun-sun/pgm_graph_inference)**

The paper discusses the utilization of Graph Neural Networks for inference in probabilistic graphical models. This repository aims to reproduce the experiments and results detailed in the paper.



## Contents

The repository is organized as follows:

- **`code/`**: Contains code for replicating the experiments and running the models.
- **`data/`**: Placeholder for any necessary data or instructions to obtain the data used in the experiments.
- **`results/`**: Holds generated results, outputs, and comparisons.


