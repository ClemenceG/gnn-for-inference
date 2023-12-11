"""

Interface for inference algorithms
Authors: kkorovin@cs.cmu.edu

"""

from inference.bp import BeliefPropagation
from inference.bp_damping import DampingBeliefPropagation
from inference.bp_nonsparse import BeliefPropagation_nonsparse
# from inference.gnn_inference import GatedGNNInference
from inference.exact import ExactInference
from inference.mcmc import GibbsSampling
from inference.bp_tree import TreeBP
from inference.mybp import MyBeliefPropagation
from inference.factor_gnn_inference import FactorGNNInference
from inference.vanilla_gnn_inference import VanillaGNNInference
from inference.mgnn_inference import MGNNInference
from inference.bpnn_inference import BPNNInference
from inference.fenbp_inference import FENBPInference
from inference.vanilla_gnn_lstm_inference import VanillaGNNLSTMInference
from inference.mgnn_lstm_inference import MGNNLSTMInference
from inference.factor_gnn_lstm_inference import FactorGNNLSTMInference
from inference.mgnn_attention_inference import MGANNInference

def get_algorithm(algo_name):
    """ Returns a constructor """
    if algo_name == 'fenbp_inference':
        return FENBPInference
    elif algo_name == 'bpnn_inference':
        return BPNNInference 
    elif algo_name == 'mgnn_inference':
        return MGNNInference
    elif algo_name == 'vanilla_gnn_inference':
        return VanillaGNNInference
    elif algo_name == "factor_gnn_inference":
        return FactorGNNInference 
    elif algo_name == "mybp":
        return MyBeliefPropagation
    elif algo_name == "bp":
        return BeliefPropagation
    elif algo_name == "bp_damping":
        return DampingBeliefPropagation
    elif algo_name == "bp_nonsparse":
        return BeliefPropagation_nonsparse
    elif algo_name == "tree_bp":
        return TreeBP
    elif algo_name == "gnn_inference":
        assert False
        # return GatedGNNInference
    elif algo_name == "exact":
        return ExactInference
    elif algo_name == "mcmc":
        return GibbsSampling
    elif algo_name == "vanilla_gnn_lstm":
        return VanillaGNNLSTMInference
    elif algo_name == "mgnn_lstm":
        return MGNNLSTMInference
    elif algo_name == "factor_gnn_lstm":
        return FactorGNNLSTMInference
    elif algo_name == "mgnn_attention":
        return MGANNInference
    else:
        raise ValueError("Inference algorithm {} not supported".format(algo_name))
