from collections import Counter
import numpy as np

from inference.core import Inference
from inference import get_algorithm
from graphical_models import construct_binary_mrf 
from scipy.stats import pearsonr
import pdb

class HamiltonianMC(Inference):

	def kinetic_energy(self, p):
		return 0.5 * p @ p.T

	def energy_function(self, x):
		return x @ self.W @ x.T - x @ self.u

	def hamiltonian(self, x, p):
		return self.kinetic_energy(p) + self.energy_function(x)

	def posterior_gradient(self, x):
		return 2* x @ self.W - self.u.T

	def leapfrog_step(self, x0, p0, step_size, num_steps):
		"""
		num_steps: number of leapfrog steps before next proposal state
		TODO: need to check
		"""
		# pdb.set_trace()
		p = p0 - 0.5 * step_size * self.posterior_gradient(x0)
		x = x0 + step_size * p

		for i in range(num_steps):
			graident = self.posterior_gradient(x)
			p -= 0.5*step_size * graident
			x += step_size * p

		p -= 0.5*step_size*self.posterior_gradient(x)

		return x, p

	def hmc(self, n, step_size, num_steps):
		x0 = np.array([1. if np.random.rand() < .5 else -1. for i in range(self.d)])
		x0 = np.reshape(x0, [1,self.d])
		p0 = np.random.normal(size=[1,self.d])
		samples = [np.copy(x0)]
		cnt = 0
		while cnt < n - 1:
			x, p = self.leapfrog_step(x0, p0, step_size, num_steps)
			# pdb.set_trace()
			orig = self.hamiltonian(x0, p0)
			curr = self.hamiltonian(x, p)
			p_accept = np.exp(orig - curr)
			# print(x, p, p_accept)
			# print(cnt)
			if p_accept > np.random.uniform():
				p0 = p
				new_sample = np.array([1. if xi > 0 else -1. for xi in x[0]])
				new_sample = np.expand_dims(new_sample, 0)
				x0 = new_sample
				# pdb.set_trace()
				samples.append(new_sample)
			cnt += 1
		# pdb.set_trace()
		return np.concatenate(samples)

	def collect_samples(self, graphs, n):
		samples = []
		for graph in graphs:
			self.W = graph.W
			self.d = graph.n_nodes
			self.u = np.reshape(graph.b, [self.d, 1])

			sample = self.hmc(n, 0.05, 0)
			samples.append(sample)

		return samples

	def run(self, graphs, n=1000):
		graphs_samples = self.collect_samples(graphs, n)
		# pdb.set_trace()
		res = []
		for samples, graph in zip(graphs_samples, graphs):
			# for each graph, compute pos and neg probs
			if self.mode == "marginal":
				# for each [:, i], compute empirical shares of -1 and 1
				binary_samples = np.where(samples < 0, 0, 1)
				pos_probs = binary_samples.mean(axis=0)
				neg_pos = np.stack([1-pos_probs, pos_probs], axis=1)
				assert neg_pos.shape == (graph.n_nodes, 2)
				res.append(neg_pos)
			elif self.mode == "map":
				cnt = Counter([tuple(row) for row in samples])
				most_freq = cnt.most_common(1)[0][0]
				res.append(most_freq)
		return res

def test_exact_against_mcmc():
    sizes = [5, 10, 15]
    n_samples = [500, 1000, 2000, 5000, 10000]
    n_trials = 100

    mcmc = HamiltonianMC("marginal")
    exact = get_algorithm("exact")("marginal")

    def get_exp_data(n_trials, n_nodes):
        graphs = []
        for trial in range(n_trials):
            graph = construct_binary_mrf("fc", n_nodes=n_nodes, shuffle_nodes=True)
            graphs.append(graph)
        return graphs

    for size in sizes:
        graphs = get_exp_data(n_trials, size)
        exact_res = exact.run(graphs)
        for n_samp in n_samples:
            mcmc_res = mcmc.run(graphs, n_samp)
            v1, v2  = [], []
            for graph_res in mcmc_res:
                v1.extend([node_res[1] for node_res in graph_res])
            for graph_res in exact_res:
                v2.extend([node_res[1] for node_res in graph_res])

            corr_mcmc = pearsonr(v1, v2)
            print("{},{}: correlation between exact and MCMC: {}".format(size, n_samp, corr_mcmc[0]))

if __name__ == '__main__':
	test_exact_against_mcmc()
	# hmmc = HamiltonianMC("map")
	# W = np.array([[0, -1, 0, 0, 0, 0, 0],
	# 		  [-1, 0, 1.5, 1, 0, 0, 0],
	# 		  [0, 1.5, 0, 0, 1.5, 2, -2],
	# 		  [0, 1, 0, 0, 0, 0, 0],
	# 		  [0, 0, 1.5, 0, 0, 0, 0],
	# 		  [0, 0, 2, 0, 0, 0, 0],
	# 		  [0, 0, -2, 0, 0, 0, 0]])
	# u = np.zeros(7)
	# from graphical_models.data_structs import BinaryMRF
	# graphs = [BinaryMRF(W, u)]
	# samples = hmmc.collect_samples(graphs, 100)
	# pdb.set_trace()
	# print(samples[0])