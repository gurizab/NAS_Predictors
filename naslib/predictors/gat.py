import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from naslib.predictors.gcn import graph_pooling
from naslib.utils.utils import AverageMeterGroup
from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor
from naslib.predictors.trees.ngb import loguniform


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def accuracy(prediction, target, scale=100.):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


class GraphAttentionLayer(nn.Module):
    """
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        :param h: (batch_zize, number_nodes, in_features)
        :param adj: (batch_size, number_nodes, number_nodes)
        :return: (batch_zize, number_nodes, out_features)
        """
        # batchwise matrix multiplication
        Wh = torch.matmul(h, self.W)  # (batch_zize, number_nodes, in_features) * (in_features, out_features) -> (batch_zize, number_nodes, out_features)
        e = self.prepare_batch(Wh)  # (batch_zize, number_nodes, number_nodes)

        # (batch_zize, number_nodes, number_nodes)
        zero_vec = -9e15 * torch.ones_like(e)

        # (batch_zize, number_nodes, number_nodes)
        attention = torch.where(adj > 0, e, zero_vec)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.softmax(attention, dim=-1)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # batched matrix multiplication (batch_size, number_nodes, out_features)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def prepare_batch(self, Wh):
        """
        with batch training
        :param Wh: (batch_zize, number_nodes, out_features)
        :return:
        """
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)

        B, N, E = Wh.shape  # (B, N, N)

        # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)

        # broadcast add (B, N, 1) + (B, 1, N)
        e = Wh1 + Wh2.permute(0, 2, 1)  # (B, N, N)
        return self.leakyrelu(e)

    def repr(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, hidden_l, dropout, alpha, nheads, nclass=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, hidden_l, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]
        self.attentions = nn.ModuleList(self.attentions)
        self.out_att = GraphAttentionLayer(hidden_l*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, batch):
        numv, adj, x = batch["num_vertices"], batch["adjacency"], batch["operations"]
        numv = numv.to(device)
        adj = adj.to(device)
        x = x.to(device)
        adj = normalize_adj(adj + torch.eye(adj.size(1), device=adj.device))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = graph_pooling(x, numv)
        return x.view(-1)


class GATPredictor(Predictor):

    def __init__(self, encoding_type='gcn', ss_type=None, hpo_wrapper=False):
        self.encoding_type = encoding_type
        if ss_type is not None:
            self.ss_type = ss_type
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {'hidden_l': 144,
                                    'batch_size': 7,
                                    'lr': 1e-4,
                                    'wd': 3e-4,
                                    'dropout': 0.1,
                                    'alpha': 1e-4,
                                    'nheads': 4}
        self.hyperparams = None

    def get_model(self, **kwargs):
        if self.ss_type == 'nasbench101':
            initial_hidden = 5
        elif self.ss_type == 'nasbench201':
            initial_hidden = 7
        elif self.ss_type == 'darts':
            initial_hidden = 9
        elif self.ss_type == 'nlp':
            initial_hidden = 8
        else:
            raise NotImplementedError()

        dropout = self.hyperparams['dropout']
        alpha = self.hyperparams['alpha']
        nheads = self.hyperparams['nheads']
        predictor = GAT(nfeat=initial_hidden, dropout=dropout, alpha=alpha, nheads=nheads)
        return predictor

    def fit(self, xtrain, ytrain, train_info=None,
            epochs=300):

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()

        hidden_l = self.hyperparams['hidden_l']
        batch_size = self.hyperparams['batch_size']
        lr = self.hyperparams['lr']
        wd = self.hyperparams['wd']

        dropout = self.hyperparams['dropout']
        alpha = self.hyperparams['alpha']
        nheads = self.hyperparams['nheads']

        # get mean and std, normlize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain_normed = (ytrain - self.mean) / self.std
        # encode data in gcn format
        train_data = []
        for i, arch in enumerate(xtrain):
            encoded = encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
            encoded['val_acc'] = float(ytrain_normed[i])
            train_data.append(encoded)
        train_data = np.array(train_data)

        self.model = self.get_model(hidden_l=hidden_l, dropout=dropout, alpha=alpha, nheads=nheads)
        data_loader = DataLoader(train_data, batch_size=batch_size,
                                 shuffle=True, drop_last=True)

        self.model.to(device)
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        self.model.train()

        for _ in range(epochs):
            meters = AverageMeterGroup()
            lr = optimizer.param_groups[0]["lr"]
            for _, batch in enumerate(data_loader):
                target = batch["val_acc"].float().to(device)
                prediction = self.model(batch)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                mse = accuracy(prediction, target)
                meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))

            lr_scheduler.step()
        train_pred = np.squeeze(self.query(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))
        return train_error

    def query(self, xtest, info=None, eval_batch_size=1000):
        test_data = np.array([encode(arch, encoding_type=self.encoding_type, ss_type=self.ss_type)
                              for arch in xtest])
        test_data_loader = DataLoader(test_data, batch_size=eval_batch_size)

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                prediction = self.model(batch)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return pred * self.std + self.mean

    def set_random_hyperparams(self):

        if self.hyperparams is None:
            params = self.default_hyperparams.copy()

        else:
            params = {
                'hidden_l': int(loguniform(64, 200)),
                'batch_size': int(loguniform(5, 32)),
                'lr': loguniform(.00001, .1),
                'wd': loguniform(.00001, .1),
                'dropout': 0.1,
                'alpha': 1e-4,
                'nheads': 4
            }

        self.hyperparams = params
        return params
