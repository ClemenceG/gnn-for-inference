from argparse import ArgumentParser
import os
import random
import shutil
from typing import List

import numpy as np

def parse_args():
    parser = ArgumentParser()

    # crucial arguments
    parser.add_argument('-r', '--root-dir', required=True, type=str,
                        help='Root directory where all graph types are located')
    parser.add_argument('-e', '--exclude', required=False, default=[], nargs="+", type=str,
                        help='exclude any graph type from combining')
    parser.add_argument('-n', '--num', required=True, type=int,
                        help='number of graphs to sample')
    parser.add_argument('-s', '--seed', default=12345, type=int, 
                        help='Random seed')
    parser.add_argument('-l', '--size', type=int,
                        help='graph sizes (i.e. number of nodes) to combine with each other')
    parser.add_argument('-m', '--save-name', type=str, required=True, 
                        help='Assigns a name to the folder of all combined samples')
    
    return parser.parse_args()


def _combination(root_dir: str, num: int, exclude_lst: List[int], size: int, save_folder_name: str):
    train_dir = os.path.join(root_dir, "train")
    save_dir = os.path.join(root_dir, "train", save_folder_name, str(size))

    types = [t for t in os.listdir(train_dir) if not t.startswith(".") and t != save_folder_name]
    num_total_types = len(types)
    remained_types = [t for t in types if t not in exclude_lst]
    num_remained_types = len(remained_types)
    print(f"{num_remained_types} out of {num_total_types} remained for graph type combination", flush=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        res = input(f"{save_dir} already exists. Do you want to override? (Y or N)")
        if res.lower() == "y":
            print(f"{save_dir} is overriding", flush=True)
        else:
            raise ValueError("Folder already exist! Please provide another save_name.")
        
    assert os.path.exists(train_dir), f"{train_dir} does not exist."
    
    num_samples = 0
    for type in remained_types:
        load_dir = os.path.join(train_dir, type, str(size))
        files = os.listdir(load_dir)
        random.shuffle(files)
        num_samples = min(len(files), num)
        print(f"\tNumber of samples exist for type {type}: {num_samples}", flush=True)
        selected = files[:num_samples]

        for s in selected:
            num_samples += 1 
            shutil.copy(os.path.join(load_dir, s), save_dir)
        
        print("\tSampling done.", flush=True)

    dst_num = len(os.listdir(save_dir))
    print(f"Total number of saved samples: {dst_num}", flush=True)

def main():
    args = parse_args() 
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Start data combination (seed {args.seed}) from {args.root_dir}, exclude {args.exclude},\n number of samples to combine each type {args.num}, graph sizes to combine {args.size}", flush=True)
    _combination(args.root_dir, args.num, args.exclude, args.size, args.save_name)


if __name__ == "__main__":
    main()