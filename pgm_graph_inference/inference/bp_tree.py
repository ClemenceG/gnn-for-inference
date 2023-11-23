"""
Sum-product algorithm for tree MRF 
Authors: lingxiao@cmu.edu
"""

from inference.core import Inference
import numpy as np 
from scipy.special import logsumexp
import random

class TreeBP(Inference):

    def _safe_norm_exp(self, logit):
        logit -= np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def run_one(self, graph):
        """ 
         Asynchronous BP using DFS over tree
        """
        # Question: is W symmetric or not ?
        if self.mode == "marginal": # not using log
            sumOp = logsumexp
        else:
            sumOp = np.max

        row, col = np.where(graph.W)
        n_V, n_E = len(graph.b), len(row)
        # create messages, using dict, log scale
        messages = {edge: np.zeros(2) for edge in zip(row,col)}
        # create neighbors   d
        neighbors = {i:[] for i in range(n_V)}
        for i,j in zip(row,col): neighbors[i].append(j)
        # randomly choose a node as root for tree
        root = random.choice(range(n_V))

        xij = np.array([[1,-1],[-1,1]])
        xi = np.array([-1, 1])

        def send_message(i,j): # send message for i to j
            in_message_prod = 0
            for k in neighbors[i]:
                if k != j: in_message_prod += messages[(k,i)]
            # calculate message from i to j 
            local_potential = graph.W[i,j]*xij + graph.b[i]*xi 
            messages[(i,j)] = sumOp(local_potential + in_message_prod, axis=1)

        def collect(i,j): # collect message for i comming from j
            in_message_prod = 0
            for k in neighbors[j]:
                if k != i: collect(j, k)
            # calculate message from j to i 
            send_message(j,i)

        def distribute(i,j): # distribute message from i to j
            # calculate message from i to j
            send_message(i,j)
            # send message from j to all k except i
            for k in neighbors[j]:
                if k != i: distribute(j,k)

        # leaves to root collect (down-to-up)
        for i in neighbors[root]:
            collect(root, i)
        # root to leaves distribute (up-to-down)
        for i in neighbors[root]:
            distribute(root, i)

        # calculate marginal or map
        probs = np.zeros([n_V, 2])
        for i in range(n_V):
            probs[i] = graph.b[i]*xi
            for j in neighbors[i]:
                probs[i] +=  messages[(j,i)]

        # normalize
        if self.mode == 'marginal':
            results = self._safe_norm_exp(probs)

        if self.mode == 'map':
            results = np.argmax(probs, axis=1)
            results[results==0] = -1

        return results

    def run(self, graphs, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph))
        return res
