import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from pydantic.typing import Literal
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from .utils import RBFExpansion

class PotNetConv(MessagePassing):

    def __init__(self, fc_features):
        super(PotNetConv, self).__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )

        return F.relu(x + self.bn(out))

    def message(self, x_i, x_j, edge_attr, index):
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))

class PotNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        ## model init
        self.config = config
        self.dim_h = config["dim_h"]
        self.rbf_min = config["rbf_min"]
        self.rbf_max = config["rbf_max"]
        self.inf_edge_features = config["inf_edge_features"]
        self.layer_num = config["layer_num"]
        self.output_feature = config["output_feature"]
        self.potentials = config["potentials"]
        self.atom_init = config["atom_init"]

        with open(self.atom_init) as fp:
            self.init_dict = json.load(fp)
        self.cgcnn_embedding = torch.zeros([108,92])
        for i in self.init_dict.keys():
            self.cgcnn_embedding[int(i)] = torch.tensor(self.init_dict[i])

        self.atom_embedding = nn.Linear(92,self.dim_h)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=self.rbf_min,
                vmax=self.rbf_max,
                bins=self.dim_h,
            ),
            nn.Linear(self.dim_h, self.dim_h),
            nn.SiLU(),
        )

        self.inf_edge_embedding = RBFExpansion(
            vmin=self.rbf_min,
            vmax=self.rbf_max,
            bins=self.inf_edge_features,
            type='multiquadric'
        )

        self.infinite_linear = nn.Linear(self.inf_edge_features, self.dim_h)

        self.infinite_bn = nn.BatchNorm1d(self.dim_h)

        self.conv_layers = nn.ModuleList(
            [
                PotNetConv(self.dim_h)
                for _ in range(self.layer_num)
            ]
        )


        self.fc = nn.Sequential(
            nn.Linear(self.dim_h, self.dim_h), ShiftedSoftplus()
        )

        self.fc_out = nn.Linear(self.dim_h, self.output_feature)

    def forward(self, data):
        """CGCNN function mapping graph to outputs."""
        # fixed edge features: RBF-expanded bondlengths
        edge_index = data.edge_index
        edge_features = self.edge_embedding(-0.75 / data.edge_attr)
        

        inf_edge_index = data.inf_edge_index
        inf_feat = sum([data.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.potentials)])
        inf_edge_features = self.inf_edge_embedding(inf_feat)
        inf_edge_features = self.infinite_bn(F.softplus(self.infinite_linear(inf_edge_features)))

        # initial node features: atom feature network...
        self.cgcnn_embedding = self.cgcnn_embedding.to(data.x.device)
        node_features = self.atom_embedding(self.cgcnn_embedding[data.x])
        edge_index = torch.cat([data.edge_index, inf_edge_index], 1)
        edge_features = torch.cat([edge_features, inf_edge_features], 0)

        for i in range(self.layer_num):
            node_features = self.conv_layers[i](node_features, edge_index, edge_features)

        features = global_mean_pool(node_features, data.batch)
        features = self.fc(features)
        return {"FormationEnergyPeratom":torch.squeeze(self.fc_out(features))}