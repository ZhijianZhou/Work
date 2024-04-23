import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ZTrainFrame.ModelZoo.Crystal_former.block import *
from torch_geometric.nn import  global_mean_pool

class NodeAttrPredict(nn.Module):
    def __init__(self,
                 dim_h,
                 out_dim,):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_h, dim_h), ShiftedSoftplus()
        )

        self.fc_out = nn.Linear(dim_h, out_dim)
    def forward(self,
                batch):
        features = global_mean_pool(batch.x, batch.batch)
        features = self.fc(features)
        return torch.squeeze(self.fc_out(features))
    
class NodePredict(nn.Module):
    def __init__(self,
                 dim_h,
                 out_dim,):
        super().__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(dim_h, dim_h), nn.Sigmoid()
        # )

        self.fc_out = nn.Linear(dim_h, out_dim)

    def forward(self,
                batch):
        # features = self.fc(batch.x)
        return self.fc_out(batch.x[batch.elements_mask])