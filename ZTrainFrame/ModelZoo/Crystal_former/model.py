from .crystalnet import*
from .head import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrystalPretrain(nn.Module):
    def __init__(self, 
                 dim_h,
                 layer_num,
                 local_gnn_type,
                 global_model_type,
                 inf_edge_feature,
                 inf_edge_feature_dim,
                 ):
        super().__init__()
        self.pretrain = CrystalNet(dim_h=dim_h,
                   layer_num = layer_num,
                   local_gnn_type=local_gnn_type,
                   global_model_type=global_model_type,
                   inf_edge_feature=inf_edge_feature,
                   inf_edge_feature_dim=inf_edge_feature_dim
                   )
        self.node_predict = NodePredict(dim_h = dim_h,
                                   out_dim = 94)
    def forward(self, batch):
        features = self.pretrain(batch)
        output = self.node_predict(features)
        return output
    
class CrystalFinetune(nn.Module):
    def __init__(self,
                 config, 
                 ):
        super().__init__()
        
        self.dim_h = config["dim_h"]
        self.layer_num = config["layer_num"]
        self.local_gnn_type = config["local_gnn_type"]
        self.global_model_type = config["global_model_type"]
        self.inf_edge_feature = config["inf_edge_feature"]
        self.inf_edge_feature_dim = config["inf_edge_feature_dim"]
        self.self_edge_feature = config["self_edge_feature"]
        self.atom_feature = config["atom_feature"]
        self.pretrain = CrystalNet(dim_h=self.dim_h,
                   layer_num = self.layer_num,
                   local_gnn_type= self.local_gnn_type,
                   global_model_type= self.global_model_type,
                   inf_edge_feature= self.inf_edge_feature,
                   self_edge_feature= self.self_edge_feature
                   )
        self.node_predict = NodeAttrPredict(dim_h = self.dim_h,
                                   out_dim = 1)
    def forward(self, batch):
        features = self.pretrain(batch)
        output = self.node_predict(features)
        return {"FormationEnergyPeratom":output}

        
        