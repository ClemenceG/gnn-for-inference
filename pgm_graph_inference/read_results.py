import numpy as np
import argparse
import pprint

# Take arguments from command line and read path
def read_args():
    """
    Take arguments from command line and read path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,
                        help='path to npy file')
    args = parser.parse_args()
    return args

# Read results from npy file
def read_results(file_path):
    """
    Read results from npy file
    """
    return np.load(file_path, allow_pickle=True).item()

if __name__ == "__main__":
    args = read_args()
    file_path = args.file_path

    # Validate if the file path is provided
    if file_path is None:
        print("Please provide a file path.")
    else:
        # Read and print the results
        results = read_results(file_path)
        print("Results:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(results)
        # print(results)