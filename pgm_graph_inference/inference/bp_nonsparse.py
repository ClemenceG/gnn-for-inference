"""

Approximate inference using Belief Propagation
Here we can rely on some existing library,
for example https://github.com/mbforbes/py-factorgraph
Authors: lingxiao@cmu.edu
         kkorovin@cs.cmu.edu
         markcheu@andrew.cmu.edu
"""
import numpy as np 
from scipy.special import logsumexp

from inference.core import Inference


class BeliefPropagation_nonsparse(Inference):
    """
    A special case implementation of BP
    for binary MRFs.
    Exact BP in tree structure only need two passes,
    LBP need multiple passes until convergene. 
    """

    def _safe_norm_exp(self, logit):
        logit -= np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob 

    def _safe_divide(self, a, b):
        '''
        Divies a by b, then turns nans and infs into 0, so all division by 0
        becomes 0.
        '''
        c = a / b
        c[c == np.inf] = 0.0
        c = np.nan_to_num(c)
        return c

    def run_one(self, graph, use_log=True, smooth=0):
        # Asynchronous BP  
        # Sketch of algorithm:
        # -------------------
        # preprocessing:
        # - sort nodes by number of edges
        # Algo:
        # - initialize messages to 1
        # - until convergence or max iters reached:
        #     - for each node in sorted list (fewest edges to most):
        #         - compute outgoing messages to neighbors
        #         - check convergence of messages

        if self.mode == "marginal": # not using log
            sumOp = logsumexp if use_log else np.sum
        else:
            sumOp = np.max
        # storage, W should be symmetric 
        max_iters = 100
        epsilon = 1e-10 # determines when to stop

        n_nodes = graph.W.shape[0]
        messages = np.zeros((n_nodes,n_nodes,2))
        x_potential = np.array([-1,1])
        for iter in range(max_iters):
            converged = True
            # save old message for checking convergence
            old_messages = messages.copy()
            # update messages
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if graph.W[i,j] != 0:
                        for x_j in range(2):
                            s = []
                            for x_i in range(2):
                                log_sum = graph.W[i,j]*x_potential[x_i]*x_potential[x_j]+graph.b[x_i]*x_potential[x_i]
                                for k in range(n_nodes):
                                    if (graph.W[i,k]!=0 and k!=j):
                                        log_sum+=messages[k,i,x_i]
                                s.append(log_sum)
                            messages[i,j,x_j] = logsumexp(s)
            error = (messages - old_messages)**2
            error = error.mean()
            if error < epsilon: break

        if self.verbose: print("Is BP converged: {}".format(converged))

        # calculate marginal
        probs = np.zeros((n_nodes,2))
        for i in range(n_nodes):
            for x_i in range(2):
                probs[i,x_i] = graph.b[i]*x_potential[x_i] # -1, 1
                for k in range(n_nodes):
                    if(graph.W[i,k]!=0):
                        probs[i,x_i] += messages[k,i,x_i]
        results = self._safe_norm_exp(probs)


        return results


    def run(self, graphs, use_log=True, verbose=False):
        self.verbose = verbose
        res = []
        for graph in graphs:
            res.append(self.run_one(graph, use_log=use_log))
        return res


if __name__ == "__main__":
    bp = BeliefPropagation_nonsparse("marginal")
    
