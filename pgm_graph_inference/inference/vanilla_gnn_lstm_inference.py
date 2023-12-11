"""

Treat messages as nodes
Defines GNNInference objects: models that perform
inference, given a graphical model.

"""

import torch
import numpy as np
from tqdm import tqdm

from inference.core import Inference
# from inference.factor_gnn_sparse import GGNN as GGNN_sparse
from inference.vanilla_gnn_lstm_sparse import VGNN_LSTM
# from inference.ggnn_model import GGNN


class VanillaGNNLSTMInference(Inference):
    def __init__(self, mode, state_dim, message_dim, 
                hidden_unit_message_dim, hidden_unit_readout_dim, 
                n_steps=10, load_path=None, sparse=True):
        Inference.__init__(self, mode)
        # self.model = GGNN(state_dim, message_dim,
                  # hidden_unit_message_dim,
                  # hidden_unit_readout_dim, n_steps) 

        assert sparse
        self.model = VGNN_LSTM(state_dim, message_dim,
                  hidden_unit_message_dim,
                  hidden_unit_readout_dim, n_steps) 

        if load_path is not None:
            self.model.load_state_dict(
                torch.load(
                    load_path,
                    map_location=lambda storage,
                    loc: storage))
            self.model.eval()
        self.history = {"loss": []}
        self.batch_size = 50

    def run_one(self, graph, device):
        """ Forward computation that depends on the mode """
        # Call to super forward
        # wrap up depending on mode 
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            out = self.model(J,b)
            return out.detach().cpu().numpy()

    def run(self, graphs, device, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph, device))
        return res

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def train(self, dataset, optimizer, criterion, device):
        """ One epoch of training """
        # TODO: set self.batch_size depending on device type
        self.model.to(device)
        self.model.train()
        self.model.zero_grad()

        batch_loss = []
        mean_losses = []

        for i, graph in tqdm(enumerate(dataset)):
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            out = self.model(J,b)

            if self.mode == "marginal":
                target = torch.from_numpy(graph.marginal).float().to(device)
                loss = criterion(out, target)
            else:
                map_bin = np.where(graph.map > 0.1, 1, 0)
                target = torch.from_numpy(map_bin).float().to(device)
                loss = criterion(out, target)  # bce

            batch_loss.append(loss)

            if (i % self.batch_size == 0):
                ll_mean = torch.stack(batch_loss).mean()
                ll_mean.backward()
                optimizer.step()
                self.model.zero_grad()
                batch_loss=[]
                mean_losses.append(ll_mean.item())
            if i > 50: break

        self.history["loss"].append(np.mean(mean_losses))
