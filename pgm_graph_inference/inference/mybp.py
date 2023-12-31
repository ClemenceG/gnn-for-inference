"""
A Belief Propagation implementation taken from https://github.com/ilyakava/sumproduct/blob/master/sumproduct.py
"""

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

from inference.core import Inference


class Node:
    def __init__(self, name):
        self.connections = []
        self.inbox = {}  # messages recieved
        self.name = name

    def append(self, to_node):
        """
    Mutates the to AND from node!
    """
        self.connections.append(to_node)
        to_node.connections.append(self)

    def deliver(self, step_num, mu):
        """
    Ensures that inbox is keyed by a step number
    """
        if self.inbox.get(step_num):
            self.inbox[step_num].append(mu)
        else:
            self.inbox[step_num] = [mu]


class Factor(Node):
    """
  NOTE: For the Factor nodes in the graph, it will be assumed
  that the connections are created in the same exact order
  as the potentials' dimensions are given
  """

    def __init__(self, name, potentials):
        self.p = potentials
        Node.__init__(self, name)

    def make_message(self, recipient):
        """
    Does NOT mutate the Factor node!

    NOTE that using the log rule before 5.1.42 in BRML by David
    Barber, that the product is actually computed via a sum of logs.

    Steps:
    1. reformat mus to all be the same dimension as the factor's
    potential and take logs, mus -> lambdas
    2. find a max_lambda (element wise maximum)
    3. sum lambdas, and subtract the max_lambda once
    4. exponentiate the previous result, multiply by exp of max_lambda
    and run summation to sum over all the states not in the recipient
    node
    5. log previous, add back max_lambda, and exponentiate because we
    will pass around mus rather than lambdas everywhere

    Note that max_lambda in 5.1.42 is NOT a element-wise maximum (and
    therefore a matrix), it is a scalar.
    """
        if not len(self.connections) == 1:
            unfiltered_mus = self.inbox[max(self.inbox.keys())]
            mus = [mu for mu in unfiltered_mus
                   if not mu.from_node == recipient]
            all_mus = [self.reformat_mu(mu) for mu in mus]
            lambdas = np.array([np.log(mu) for mu in all_mus])
            max_lambdas = np.nan_to_num(lambdas.flatten())
            max_lambda = max(max_lambdas)
            result = sum(lambdas) - max_lambda
            product_output = np.multiply(self.p, np.exp(result))
            return np.exp(
                np.log(self.summation(product_output, recipient)) + max_lambda)
        else:
            return self.summation(self.p, recipient)

    def reformat_mu(self, mu):
        """
    Returns the given mu's val reformatted to be the same
    dimensions as self.p, ensuring that mu's values are
    expanded in the correct axes.

    The identity of mu's from_node is used to decide which axis
    the mu's val should be expaned in to fit self.p

    Example:

    # self.p (dim order: x3, x4, then x2)
    np.array([
      [
        [0.3,0.5,0.2],
        [0.1,0.1,0.8]
      ],
      [
        [0.9,0.05,0.05],
        [0.2,0.7,0.1]
      ]
    ])

    # mu
    x3 = np.array([0.2, 0.8])
    which_dim = 0 # the dimension which x3 changes in self.p
    dims = [2, 2, 3]

    # desired output
    np.array([
      [
        [0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2]
      ],
      [
        [0.8, 0.8, 0.8],
        [0.8, 0.8, 0.8]
      ]
    ])
    """
        dims = self.p.shape
        states = mu.val
        which_dim = self.connections.index(mu.from_node)  # raises err
        assert dims[which_dim] is len(states)

        acc = np.ones(dims)
        for coord in np.ndindex(dims):
            i = coord[which_dim]
            acc[coord] *= states[i]
        return acc

    def summation(self, p, node):
        """
    Does NOT mutate the factor node.

    Sum over all states not in the node.
    Similar to reformat_mu in strategy.
    """
        dims = p.shape
        which_dim = self.connections.index(node)
        out = np.zeros(node.size)
        assert dims[which_dim] is node.size
        for coord in np.ndindex(dims):
            i = coord[which_dim]
            out[i] += p[coord]
        return out


class Variable(Node):
    def __init__(self, name, size):
        self.bfmarginal = None
        self.size = size
        Node.__init__(self, name)

    def marginal(self):
        """
    Life saving normalizations:

    sum_logs - max(sum_logs) <- before exponentiating
    and rem_inf
    """
        if len(self.inbox):
            mus = self.inbox[max(self.inbox.keys())]
            log_vals = [np.log(mu.val) for mu in mus]
            valid_log_vals = [np.nan_to_num(lv) for lv in log_vals]
            sum_logs = sum(valid_log_vals)
            valid_sum_logs = sum_logs - max(sum_logs)  # IMPORANT!
            prod = np.exp(valid_sum_logs)
            return prod / sum(prod)  # normalize
        else:
            # first time called: uniform
            return np.ones(self.size) / self.size

    def latex_marginal(self):
        """
    same as marginal() but returns a nicely formatted latex string
    """
        data = self.marginal()
        data_str = ' & '.join([str(d) for d in data])
        tabular = '|' + ' | '.join(['l' for i in range(self.size)]) + '|'
        return ("$$p(\mathrm{" + self.name + "}) = \\begin{tabular}{" + tabular
                + '} \hline' + data_str + '\\\\ \hline \end{tabular}$$')

    def make_message(self, recipient):
        """
    Follows log rule in 5.1.38 in BRML by David Barber
    b/c of numerical issues
    """
        if not len(self.connections) == 1:
            unfiltered_mus = self.inbox[max(self.inbox.keys())]
            mus = [mu for mu in unfiltered_mus
                   if not mu.from_node == recipient]
            log_vals = [np.log(mu.val) for mu in mus]
            return np.exp(sum(log_vals))
        else:
            return np.ones(self.size)


class Mu:
    """
  An object to represent a message being passed
  a to_node attribute isn't needed since that will be clear from
  whose inbox the Mu is sitting in
  """

    def __init__(self, from_node, val):
        self.from_node = from_node
        # this normalization is necessary
        self.val = val.flatten() / sum(val.flatten())


class FactorGraph:
    def __init__(self, first_node=None, silent=False, debug=False):
        self.nodes = {}
        self.silent = silent
        self.debug = debug
        if first_node:
            self.nodes[first_node.name] = first_node

    def add(self, node):
        assert node not in self.nodes
        self.nodes[node.name] = node

    def connect(self, name1, name2):
        # no need to assert since dict lookup will raise err
        self.nodes[name1].append(self.nodes[name2])

    def append(self, from_node_name, to_node):
        assert from_node_name in self.nodes
        tnn = to_node.name
        # add the to_node to the graph if it is not already there
        if not (self.nodes.get(tnn, 0)):
            self.nodes[tnn] = to_node
        self.nodes[from_node_name].append(self.nodes[tnn])
        return self

    def leaf_nodes(self):
        return [node for node in self.nodes.values()
                if len(node.connections) == 1]

    def observe(self, name, state):
        """
    Mutates the factors connected to Variable with name!

    @param state: Ordinal state starting at ONE (1)

    As described in Barber 5.1.3. But instead of multiplying
    factors with an indicator/delta_function to account for
    an observation, the factor node loses the dimensions for
    unobserved states, and then the connection to the observed
    variable node is severed (although it remains in the graph
    to give a uniform marginal when asked).
    """
        assert False
        node = self.nodes[name]
        assert isinstance(node, Variable)
        assert node.size >= state
        assert state, "state is obsered on an ordinal scale starting at ONE(1)"
        for factor in [c for c in node.connections if isinstance(c, Factor)]:
            delete_axis = factor.connections.index(node)
            delete_dims = list(range(node.size)) # Fixed because pop is not a method of the range object
            delete_dims.pop(state - 1)
            sliced = np.delete(factor.p, delete_dims, delete_axis)
            factor.p = np.squeeze(sliced)
            factor.connections.remove(node)
            assert len(factor.p.shape) is len(factor.connections)
        node.connections = []  # so that they don't pass messages

    def export_marginals(self):
        return dict([
            (n.name, n.marginal()) for n in self.nodes.values()
            if isinstance(n, Variable)
        ])

    @staticmethod
    def compare_marginals(m1, m2):
        """
    For testing the difference between marginals across a graph at
    two different iteration states, in order to declare convergence.
    """
        assert not len(np.setdiff1d(m1.keys(), m2.keys()))
        return sum([sum(np.absolute(m1[k] - m2[k])) for k in m1.keys()])

    def compute_marginals(self, max_iter=500, tolerance=1e-6, error_fun=None):
        """
    sum-product algorithm

    @param error_fun: a custom error function that takes two arguments
    each of the form of export_marginals' return value

    @return epsilons[1:]: if you are using the default error function
    compate_marginals, the first two epsilons are meaningless (first
    entry is arbitrarily 1, and first marginal is arbitrarily uniform
    so the second computed epsilon will also be arbitrary). HOWEVER,
    those using a custom error function might only using the most
    recently computed marginal, and would be interested in epsilons[1].

    Mutates nodes by adding in the messages passed into their
    'inbox' instance variables. It does not change the potentials
    on the Factor nodes.

    Using the "Asynchronous Parallel Schedule" from Sudderth lec04
    slide 11 after an initialization step of Variable nodes sending
    all 1's messages:
    - At each iteration, all nodes compute all outputs from all
    current inputs. Factors-Variables and then Variables-Factors
    - Iterate until convergence.

    This update schedule is best suited for loopy graphs. It ends
    up working best as a max sum-product algorithm as high
    probabilities dominate heavily when the tolerance is very small
    """
        # for keeping track of state
        epsilons = [1]
        step = 0
        # for inbox clearance
        for node in self.nodes.values():
            node.inbox.clear()
            node.bfmarginal = None
        # for testing convergence
        cur_marginals = self.export_marginals()
        # initialization
        for node in self.nodes.values():
            if isinstance(node, Variable):
                message = Mu(node, np.ones(node.size))
                for recipient in node.connections:
                    recipient.deliver(step, message)

        # propagation (w/ termination conditions)
        while (step < max_iter) and tolerance < epsilons[-1]:
            last_marginals = cur_marginals
            step += 1
            if not self.silent:
                epsilon = 'epsilon: ' + str(epsilons[-1])
                print(epsilon + ' | ' + str(step) + '-' * 20)
            factors = [n for n in self.nodes.values() if isinstance(n, Factor)]
            variables = [n for n in self.nodes.values()
                         if isinstance(n, Variable)]
            senders = factors + variables
            for sender in senders:
                next_recipients = sender.connections
                for recipient in next_recipients:
                    if self.debug:
                        print(sender.name + ' -> ' + recipient.name)
                    val = sender.make_message(recipient)
                    message = Mu(sender, val)
                    recipient.deliver(step, message)
            cur_marginals = self.export_marginals()
            if error_fun:
                epsilons.append(error_fun(cur_marginals, last_marginals))
            else:
                epsilons.append(
                    self.compare_marginals(cur_marginals, last_marginals))
        if not self.silent:
            print('X' * 50)
            print('final epsilon after ' + str(step) + ' iterations = ' + str(
                epsilons[-1]))
        return epsilons[1:]  # skip only the first, see docstring above

    def brute_force(self):
        """
    Main strategy of this code was gleaned from:
    http://cs.brown.edu/courses/cs242/assignments/hw1code.zip

    # first compute the full joint table
    - create a joint accumulator for N variables that is N dimensional
    - iterate through factors
      - for each factor expand probabilities into dimensions of the joint
      table
        - create a factor accumulator that is N dimensional
        - for each coord in the joint table, look at the states of the
        vars that are in the factor's potentials, and add in the log of
        that probability
    - exponentiate and normalize
    # then compute the marginals
    - iterate through variables
      - for each variable sum over all other variables
    """
        def _safe_norm_exp(logit):
            logit -= np.max(logit, keepdims=True)
            prob = np.exp(logit)
            prob /= prob.sum(keepdims=True)
            return prob

        variables = [v for v in self.nodes.values() if isinstance(v, Variable)]
        # for v in variables: assert int(v.name) == variables.index(v)

        var_dims = [v.size for v in variables]
        N = len(var_dims)
        assert N < 32, "max number of vars for brute force is 32 (numpy's matrix dim limit)"

        log_joint_acc = np.zeros(var_dims)
        for factor in [f for f in self.nodes.values()
                       if isinstance(f, Factor)]:
            # dimensions that will matter for this factor
            which_dims = [variables.index(v) for v in factor.connections]
            factor_acc = np.ones(var_dims)
            for joint_coord in np.ndindex(tuple(var_dims)):
                factor_coord = tuple([joint_coord[i] for i in which_dims])
                factor_acc[joint_coord] *= factor.p[factor_coord]
            log_joint_acc += np.log(factor_acc)

        joint_acc = _safe_norm_exp(log_joint_acc)
        for i, variable in enumerate(variables):
            axes = tuple(j for j in range(N) if j != i)
            collapsing_marginal = joint_acc.sum(axis=axes)

            variable.bfmarginal = collapsing_marginal
        return variables


def to_factor_graph(graph):
    n_nodes = graph.W.shape[0]
    g = FactorGraph(silent=True)
    variables = [Variable('{}'.format(i), 2) for i in range(n_nodes)]

    number_of_factors = 0
    for i in range(n_nodes):
        factor_name =  'f{}'.format(i)
        f_ = Factor(factor_name, np.array([np.exp(-graph.b[i]), np.exp(graph.b[i])]))
        g.add(f_)
        g.append(factor_name, variables[i])
        number_of_factors += 1

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if graph.W[i, j] != 0.:
                factor_name = 'f{}{}'.format(i,j)
                fij = Factor(factor_name, np.array([
                    [np.exp((graph.W[i,j] + graph.W[j,i])/2), np.exp(-graph.W[i,j])],
                    [np.exp(-graph.W[j,i]), np.exp((graph.W[i,j]+ graph.W[j, i])/2)],
                ]))
                # fij = Factor(factor_name, np.array([
                    # [np.exp(graph.W[i,j] + graph.W[j,i]), np.exp(-2*graph.W[i,j])],
                    # [np.exp(-2*graph.W[j,i]), np.exp(graph.W[i,j]+ graph.W[j, i])],
                # ]))
                g.add(fij)
                g.append(factor_name, variables[i])
                g.append(factor_name, variables[j])
                # g.append(factor_name, variables[i])
                # g.append(factor_name, variables[j])
                number_of_factors += 1

    m = int(np.sqrt(n_nodes))
    assert number_of_factors == n_nodes + (m-1)*m*2
    return g

class MyBeliefPropagation(Inference):
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

        # TODO: check more convergence conditions, like calibration
        if self.mode == "marginal": # not using log
            sumOp = logsumexp if use_log else np.sum
        else:
            sumOp = np.max

        n_nodes = graph.W.shape[0]
        g = to_factor_graph(graph)
        g.compute_marginals(max_iter=200, tolerance=1e-20)
        results = probs = np.array([g.nodes['{}'.format(i)].marginal() for i in range(n_nodes)])
        # print(graph.W)
        # print(results)
        # input()
        # probs should be `
        # normalize

        # if self.mode == 'marginal':
            # if use_log:
                # results = self._safe_norm_exp(probs)
            # else:
                # results = self._safe_divide(probs, probs.sum(axis=1, keepdims=True))

        if self.mode == 'map':
            results = np.argmax(probs, axis=1)
            results[results==0] = -1

        return results


    def run(self, graphs, use_log=True, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in tqdm(graph_iterator):
            res.append(self.run_one(graph, use_log=use_log))
        return res


if __name__ == "__main__":
    # bp = BeliefPropagation("marginal")

    g = FactorGraph(silent=True) # init the graph without message printouts
    x1 = Variable('x1', 2) # init a variable with 2 states
    x2 = Variable('x2', 2) # init a variable with 3 states
    x3 = Variable('x3', 2) # init a variable with 3 states
    f12 = Factor('f12', np.array([
      [0.8,2],
      [0.2,8],
    ])) # create a factor, node potential for p(x1 | x2)
    # connect the parents to their children
    g.add(f12)
    g.append('f12', x2) # order must be the same as dimensions in factor potential!
    g.append('f12', x1) # note: f12 potential's shape is (3,2), i.e. (x2,x1)

    f12 = Factor('f23', np.array([
      [8,1],
      [2,0.8],
    ])) # create a factor, node potential for p(x1 | x2)
    # connect the parents to their children
    g.add(f12)
    g.append('f23', x2) # order must be the same as dimensions in factor potential!
    g.append('f23', x3) # note: f12 potential's shape is (3,2), i.e. (x2,x1)

    f12 = Factor('f31', np.array([
      [0.8,1],
      [2, .8],
    ])) # create a factor, node potential for p(x1 | x2)
    # connect the parents to their children
    g.add(f12)
    g.append('f31', x3) # order must be the same as dimensions in factor potential!
    g.append('f31', x1) # note: f12 potential's shape is (3,2), i.e. (x2,x1)

    g.brute_force()
    print(g.nodes['x1'].bfmarginal)
    print(g.nodes['x2'].bfmarginal)
    g.compute_marginals()
    print(g.nodes['x1'].marginal())
    print(g.nodes['x2'].marginal())


