import torch
import torch_geometric.nn.models 
from torch import nn
import torch
from ZTrainFrame.ModelZoo.eScN.escn import eSCN

class escn(nn.Module):
    def __init__(self,
                 config, 
                 ):
        super().__init__()

        self.model = eSCN(
            regress_forces = config["regress_forces"],
            otf_graph = config["otf_graph"],
            max_neighbors = config["max_neighbors"],
            cutoff = config["cutoff"],
            num_layers = config["num_layers"],
            sphere_channels = config["sphere_channels"],
            hidden_channels = config["hidden_channels"],
            lmax_list  = config["lmax_list"],
            mmax_list = config["mmax_list"],
            num_sphere_samples = config["num_sphere_samples"],
            distance_function = config["distance_function"],
            basis_width_scalar = config["basis_width_scalar"],
            max_num_elements= config["max_num_elements"],
        )
    def forward(self, batch):

        output = self.model(batch)
        return {"energy":output["energy"].squeeze(-1)}