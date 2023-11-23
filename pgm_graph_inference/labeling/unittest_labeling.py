"""
Unit tests for approximate labeling

NOTE:
* According to tests, Louvain is a reasonable choice.
"""

import unittest
import numpy as np
from time import time

from labeling import LabelProp, LabelTree, LabelSG
from graphical_models import construct_binary_mrf
from inference import get_algorithm

class TestInference(unittest.TestCase):
    def setUp(self):
        self.graph_star = construct_binary_mrf("star", n_nodes=10,
                                        shuffle_nodes=False)
        self.graph_fc = construct_binary_mrf("fc", n_nodes=10,
                                            shuffle_nodes=False)
        self.graph_barbell_100 = construct_binary_mrf("barbell", n_nodes=100,
                                        shuffle_nodes=False)
        self.graph_cycle_100 = construct_binary_mrf("cycle", n_nodes=100,
                                            shuffle_nodes=False)
    
    def run_sg_with_method(self, graph, algorithm, verbose):
        exact = get_algorithm("exact")("marginal")
        mcmc = get_algorithm("mcmc")("marginal")

        labelSG = LabelSG(algorithm=algorithm, inf_algo=mcmc)
        true_res = mcmc.run([graph])
        t0 = time()
        res = labelSG.run([graph], verbose=verbose)
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"Partition {algorithm} MSE error: \t{mse_err} in \t{time()-t0} seconds")

    def run_lbp_subgraph(self,graph,verbose=False):
        self.run_sg_with_method(graph, 'louvain', verbose)
        self.run_sg_with_method(graph, 'girvan-newman', verbose)
        self.run_sg_with_method(graph, 'igraph-community-infomap', verbose)
        self.run_sg_with_method(graph, 'igraph-label-propagation', verbose)
        # self.run_sg_with_method(graph, 'igraph-optimal-modularity', verbose)

    def run_lbp_on_graph(self, graph):
        exact = get_algorithm("exact")("marginal")

        print("With subgraph of size 1")
        lbp = LabelProp([1], exact)
        res = lbp.run([graph])
        true_res = exact.run([graph])
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"MSE error: {mse_err}")

        print("With subgraph of size 5")
        lbp = LabelProp([5], exact)
        res = lbp.run([graph])
        true_res = exact.run([graph])
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"MSE error: {mse_err}")

        print("With subgraph of size 10")
        lbp = LabelProp([10], exact)
        res = lbp.run([graph])
        true_res = exact.run([graph])
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"MSE error: {mse_err}")

    def run_tree_on_graph(self, graph):
        lbt = LabelTree("marginal")
        res = lbt.run([graph])
        true_res = exact.run([graph])
        mse_err = np.sqrt(np.sum(np.array(res) - np.array(true_res))**2)
        print(f"MSE error: {mse_err}")

    def _test_label_prop(self):
        """ Testing marginal label_prop """
        self.run_lbp_on_graph(self.graph_star)
        self.run_lbp_on_graph(self.graph_fc)

    def _test_tree_prop(self):
        """ Testing tree-based generation """
        print("Trees:")
        self.run_tree_on_graph(self.graph_star)
        self.run_tree_on_graph(self.graph_fc)

    def test_graph_cut(self):
        """ Testing graph cut """
        print('barbell')
        self.run_lbp_subgraph(self.graph_barbell_100,True)
        print('graph cycle')
        self.run_lbp_subgraph(self.graph_cycle_100,True)


if __name__ == "__main__":
    unittest.main()
