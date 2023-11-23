"""
Approximate labeling class with Label Propagation.

A subgraph is chosen and labeled using a passed algorithm,
then labels are propagated to the rest of the graph.

@author: kkorovin@cs.cmu.edu

TODO:
* bad normalization of probabilities;
  possible solutions:
  - softmax with temperature (T=1 => too uniform)
  - choose weights somehow wisely (positive weights => no problems)
"""

import numpy as np
import networkx as nx
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # shape (n_nodes, n_classes)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_softmax(x):
    # shape (n_nodes, n_classes)
    print("Logprobs:", x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    res = np.log(e_x / e_x.sum(axis=1, keepdims=True))
    assert np.all(~np.isnan(res))
    return res


class LabelProp:
    """ Adapted from LabelPropagation in sklearn """
    def __init__(self, sg_sizes, inf_algo, method='neg_label',
                 max_iter=50, tol=1e-3, T=0.2):
        """Constructor.
        Arguments:
            sg_sizes {list[int]} -- list of subgraph sizes to sample
                                    for running inf_algo on
            inf_algo {Inference} -- Inference object to run on subgraph
            method {str} -- propagation method (softmax_T, split_signs,
                            neg_label, or default)
            max_iter {int} -- max number of propagation iterations
            tol {float} -- early stopping criterion
        """
        self.sg_sizes  = sg_sizes
        self.inf_algo = inf_algo  # already knows about the mode
                                  # (marginal/map)
        self.method = method
        # set label prop params:
        self.max_iter = max_iter
        self.tol = tol
        self.T = T  # for softmax methods
        self.n_iter_ = 0

    def run_one(self, graph):
        n_nodes, n_classes = len(graph.W), 2

        self.label_distributions_ = np.ones((n_nodes, n_classes)) * 1/n_classes
        rem_nodes = list(range(n_nodes))

        # HERE
        # labprop_labels = []

        for sz in self.sg_sizes:
            # 1. sample sz nodes from remaining nodes
            n_to_sample = min(sz, len(rem_nodes))
            sampled_nodes = np.random.choice(rem_nodes, n_to_sample, replace=False)
            sampled_nodes = sorted(sampled_nodes)
            # 2. build subgraph graphical model:
            sg = graph.get_subgraph_on_nodes(sampled_nodes)

            # 3. label and record it
            labeled_distr = self.inf_algo.run([sg])[0]
            self.label_distributions_[sampled_nodes] = labeled_distr

            # 4. remove nodes
            rem_nodes = [nd for nd in rem_nodes if nd not in sampled_nodes]

            # HERE
            # labprop_labels.extend(list(l[1] for l in labeled_distr))
        # plt.hist(labprop_labels, bins=100, density=True)
        # plt.show()

        graph_matrix = self._get_graph_matrix(graph.W)
        if self.method in ['split_signs', 'neg_label']:
            graph_matrix_pos = np.where(graph_matrix > 0, graph_matrix, 0)
            graph_matrix_neg = np.where(graph_matrix < 0, -graph_matrix, 0)
        unlabeled = np.array([(i in rem_nodes) for i in range(n_nodes)])  #(y == -1)

        y_static = np.copy(self.label_distributions_)
        y_static[unlabeled] = 0

        l_previous = np.zeros((n_nodes, n_classes))
        unlabeled = unlabeled[:, np.newaxis]
 
        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break

            l_previous = self.label_distributions_

            if self.method == 'softmax_T':
                self.label_distributions_ = softmax(np.dot(graph_matrix, self.label_distributions_)/self.T)

            elif self.method == 'split_signs':
                label_distributions_pos = np.dot(graph_matrix_pos, self.label_distributions_)
                label_distributions_neg = np.dot(graph_matrix_neg, self.label_distributions_)
                self.label_distributions_ = label_distributions_pos - label_distributions_neg
                self.label_distributions_ -= self.label_distributions_.min() - 1e-10
                normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
                self._check_normalizer(normalizer)
                self.label_distributions_ /= normalizer

            elif self.method == 'neg_label':
                label_distributions_pos = np.dot(graph_matrix_pos, self.label_distributions_[:, 0])
                label_distributions_neg = np.dot(graph_matrix_neg, self.label_distributions_[:, 1])
                # self.label_distributions_[:, 0] = 0.5 * (label_distributions_pos + (1 - label_distributions_neg))
                # self.label_distributions_[:, 1] = 0.5 * (label_distributions_neg + (1 - label_distributions_pos))
                self.label_distributions_[:, 0] = sigmoid(label_distributions_pos - label_distributions_neg)
                self.label_distributions_[:, 1] = sigmoid(label_distributions_neg - label_distributions_pos)
                normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
                self._check_normalizer(normalizer)
                self.label_distributions_ /= normalizer

            elif self.method == 'default':
                self.label_distributions_ = np.dot(graph_matrix, self.label_distributions_)
                normalizer = np.sum(
                    self.label_distributions_, axis=1)[:, np.newaxis]
                self._check_normalizer(normalizer)
                self.label_distributions_ /= normalizer

            else:
                raise ValueError(f"Propagation method {self.method} not implemented.")
            
            self.label_distributions_ = np.where(unlabeled,
                                                self.label_distributions_,
                                                y_static)

        else:
            warnings.warn(f'max_iter={self.max_iter} was reached without convergence.')
            self.n_iter_ += 1

        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        self._check_normalizer(normalizer)
        self.label_distributions_ /= normalizer
        self._check_output(self.label_distributions_)
        return self.label_distributions_

    def run(self, graphs, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph))
        if self.verbose:
            # for a sanity check, plot histogram
            labprop_labels = []
            for graph_res in res:
                labprop_labels.extend(list(m[1] for m in graph_res))
            plt.hist(labprop_labels, bins=100, density=True)
            plt.show()
        return res

    def _get_graph_matrix(self, W):
        return W
        # return np.abs(W)

    def _check_normalizer(self, normalizer):
        assert np.all(normalizer != 0.), normalizer

    def _check_output(self, output):
        assert np.all(output >= 0.) and np.all(output <= 1.), output
