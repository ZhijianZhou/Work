import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import *
def generate_padding_matrix(N):
    max_length = N.shape[0]
    padding_matrix = torch.zeros((max_length, 512))
    for i, length in enumerate(N):
        padding_matrix[i, :length] = 1.0
    return padding_matrix

class CrystalNet(nn.Module):
    def __init__(self, 
                 dim_h,
                 layer_num,
                 local_gnn_type,
                 global_model_type,
                 inf_edge_feature,
                 self_edge_feature,
                 atom_feature = "atomic_number"
                 ):
        super().__init__()
        self.self_edge_feature = self_edge_feature
        if atom_feature == "atomic_number":
            self.atom_embedding = Embedding(num_embeddings=103,
                                        embedding_dim=dim_h,
                                        padding_idx=0)
        elif atom_feature == "cgcnn":
            self.atom_embedding = nn.Linear(92,dim_h)
        
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0,vmax=5.0,bins=40,type="gaussian"),
            nn.Linear(40,dim_h),
            nn.SiLU(),
        )
        self.inf_edge_feature = inf_edge_feature
        self.potentials = [-0.801, -0.074, 0.145] 
        
        self.inf_edge_embedding = RBFExpansion(
            vmin=-5.0,
            vmax=5.0,
            bins=40,
            type="gaussian"
        )

        self.infinite_linear = nn.Linear(40, dim_h)

        self.infinite_bn = nn.BatchNorm1d(dim_h)

        layers = []
        
        for _ in range(0,layer_num):
            layers.append(CrystalLayer(
                                dim_h=dim_h,
                                local_gnn_type=local_gnn_type,
                                global_model_type=global_model_type,
                                num_heads=8,
                                equivstable_pe=False, 
                                dropout=0.05,
                                attn_dropout=0.05,
                                layer_norm=True, 
                                batch_norm=False,
                                log_attn_weights=False,
                                inf_edge_feature=self.inf_edge_feature,
                                self_edge_feature=self.self_edge_feature
                                ))
        
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, batch):
        batch.x = self.atom_embedding(batch.x)
        batch.edge_attr = self.edge_embedding(batch.edge_attr)
        inf_feat = sum([batch.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.potentials)])
        inf_edge_features = self.inf_edge_embedding(inf_feat)
        batch.inf_edge_attr = self.infinite_bn(F.softplus(self.infinite_linear(inf_edge_features)))

        edge_index = torch.cat([batch.edge_index, batch.inf_edge_index], 1)
        edge_features = torch.cat([batch.edge_attr, batch.inf_edge_attr], 0)
        batch.edge_index = edge_index
        batch.edge_attr = edge_features
        for module in self.layers:
            batch = module(batch)

        return batch
        