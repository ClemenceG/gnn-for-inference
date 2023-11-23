"""
Subgraph labeling algorithm:
splits a graph into k subgraphs and labels each individually.

@author: markcheu@andrew.cmu.edu,
         kkorovin@cs.cmu.edu

NOTE:
* if the algorithm is `exact`,
  the communities with size >= 20 are unmanageable;
  currently run_one() splits such communities into chunks of 20
  *by index*, which is most likely suboptimal
  (TODO: community finding?)
"""

from inference import ExactInference

from sklearn.cluster import spectral_clustering
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_community
import matplotlib.pyplot as plt
import community as com
from tqdm import tqdm

# requires installing igraph from https://igraph.org/python/#startpy:
import igraph as ig


class LabelSG:
    def __init__(self, inf_algo, algorithm, unweighted=False):
        self.inf_algo = inf_algo
        self.algorithm = algorithm
        self.unweighted = unweighted
        self.max_subgraph_size = 15  # for exact inf_algo, do not make this large

    def run_one(self, graph, verbose):
        partition = self.partition_graph(graph, verbose)
        result = np.zeros((graph.n_nodes, 2))
        # group by partitionID and extract subgraphs to label
        for i in set(partition.values()):
            nodes_in_partition = [k for (k, v) in partition.items() if v ==  i]
            subgraph = graph.get_subgraph_on_nodes(nodes_in_partition)
            
            if len(nodes_in_partition) <= 20 or (not isinstance(self.inf_algo, ExactInference)):
                subgraph_res = self.inf_algo.run([subgraph])[0]
            else:
                # if a community is too big and we are using Exact inference, split into chunks
                subgraph_res = np.zeros((len(nodes_in_partition), 2))
                for i in range(0, len(nodes_in_partition), self.max_subgraph_size):
                    n_list = nodes_in_partition[i:i+self.max_subgraph_size]
                    subsubgraph = graph.get_subgraph_on_nodes(n_list)
                    subgraph_res[i:i+self.max_subgraph_size] = self.inf_algo.run([subsubgraph])[0]
            
            result[nodes_in_partition] = subgraph_res
        assert np.allclose(result.sum(axis=1), 1, 1e-5), result.sum(axis=1)
        return result

    def run(self, graphs, verbose=False):
        res = []
        graph_iterator = tqdm(graphs) if verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph, verbose=verbose))
        return res

    # from https://github.com/taynaud/python-louvain
    def partition_graph(self, graph, verbose=False):
        N = graph.n_nodes
        adj2 = graph.W
        adj = np.copy(adj2)
        nx_g = nx.Graph()  # networkx
        nx_g_unweighted = nx.Graph()

        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if(adj[i,j]!=0):
                    nx_g.add_weighted_edges_from([(i,j,adj[i,j])])
                    nx_g_unweighted.add_edge(i,j)
        
        if self.unweighted:
            nx_g = nx_g_unweighted

        # convert to Igraph to use their community detection algorithms.
        ig_g = ig.Graph(len(nx_g), list(zip(*list(zip(*nx.to_edgelist(nx_g)))[:2])))
        ig_g_unweighted = ig.Graph(len(nx_g_unweighted), list(zip(*list(zip(*nx.to_edgelist(nx_g)))[:2])))

        if verbose:
            print('Adjacency matrix: ', adj)

        if self.algorithm == 'edge-between':
            n_cl = N // 15
            community = ig_g.community_edge_betweenness(clusters=n_cl).as_clustering(n_cl)
            partition = self.partition_to_dict2(nx_g, community)
            if verbose:
                print('Partition by community community-edge-betweenness ', partition)

        elif self.algorithm == 'louvain':
            partition = com.best_partition(nx_g_unweighted,resolution=1)
            if verbose:
                print('Partition by Louvain Algorithm: ', partition)
                # self.visualize_partition(nx_g_unweighted,partition)

        elif self.algorithm=='girvan-newman':
            communities_generator = nx_community.girvan_newman(nx_g)
            top_level_communities = next(communities_generator)
            next_level_communities = next(communities_generator)
            partition_unorganized = sorted(map(sorted, next_level_communities))
            partition = self.partition_to_dict(nx_g,partition_unorganized)
            if verbose:
                print('Partition by Girvan-Newman Algorithm: ', partition)

        elif self.algorithm == 'igraph-community-infomap':
            community = ig_g.community_infomap()
            partition = self.partition_to_dict2(nx_g,community)
            if verbose:
                print('Partition by community infomap ', partition)

        elif self.algorithm == 'igraph-label-propagation':            
            community = ig_g.community_label_propagation()
            partition = self.partition_to_dict2(nx_g,community)
            if verbose:
                print('Partition by label propagation ', partition)

        elif self.algorithm == 'igraph-optimal-modularity':
            if nx_g.number_of_nodes() > 50:
                raise ValueError('Too many nodes for the method.')
            community = ig_g.community_optimal_modularity()  # for small graph
            partition = self.partition_to_dict2(nx_g,community)
            if verbose:
                print('Partition by integer programming (optimal modularity)', partition)
        
        elif self.algorithm=='test':
            print('GSP approach')

        # for other choices, see https://igraph.org/python/doc/python-igraph.pdf, 
        # https://yoyoinwanderland.github.io/2017/08/08/Community-Detection-in-Python/
        
        else:
            raise NotImplementedError("Partition {} not implemented yet.".format(self.algorithm))

        return partition

    # Visualize Partition
    def visualize_partition(self, graph, partition):
        size = float(len(set(partition.values())))
        pos = nx.spring_layout(graph)
        count = 0.
        for com in set(partition.values()) :
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                                        if partition[nodes] == com]
            nx.draw_networkx_nodes(graph, pos, list_nodes, node_size=20,
                                   node_color = str(count / size))

        nx.draw_networkx_edges(graph, pos, alpha=0.5)
        plt.show()

    # Visualize Adjacency Matrix
    def plot_adj(self, adj):
        plt.imshow(adj, interpolation='none')
        plt.colorbar()
        plt.show()

    # Convert Partition to Dictionary format (from network community): node id: partiiton 
    def partition_to_dict(self, graph,partition_unorganized):
        count = 0
        partition = {}
        for i in range(len(partition_unorganized)):
            partition_i = partition_unorganized[i]
            for j in range(len(partition_i)):
                partition[partition_i[j]]=count
            count += 1
        return partition

    # Convert Partition to Dictionary format (from igraph): node id: partiiton 
    def partition_to_dict2(self, graph,partition_unorganized):
        count = 0
        partition = {}
        for i in range(len(partition_unorganized)):
            partition_i = partition_unorganized[i]
            for j in range(len(partition_i)):
                partition[partition_i[j]]=count
            count += 1
        return partition

