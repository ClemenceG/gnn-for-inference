"""
Tree-based labeling
@author: kkorovin@cs.cmu.edu
"""

from tqdm import tqdm
from inference import get_algorithm

class LabelTree:
    def __init__(self, mode):
        self.inf_algo = get_algorithm("bp")(mode)

    def run_one(self, graph):
        # extract MST with the same node order
        tree = graph.get_max_abs_spanning_tree()
        # label the tree
        labels = self.inf_algo.run([tree])[0]
        return labels

    def run(self, graphs, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph))
        return res
