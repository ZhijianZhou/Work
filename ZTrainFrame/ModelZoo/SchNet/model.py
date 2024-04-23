import torch_geometric.nn.models 
from torch import nn
import torch
class SchNet(nn.Module):
    def __init__(self,
                 config, 
                 ):
        super().__init__()
        if config["atomref"]:
            
            atomrefs =  [
            -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
            -2713.48485589
        ]
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs)
            atomref = out
        else:
            atomref = torch.zeros(100)

        self.model = torch_geometric.nn.models.SchNet(
            hidden_channels=config["hidden_channels"],
            num_filters=config["num_filters"],
            num_interactions=config["num_interactions"],
            num_gaussians=config["num_gaussians"],
            cutoff=config["cutoff"],
            dipole=False,
            atomref=atomref.view(-1,1),
        )
    def forward(self, batch):

        output = self.model(batch.x,batch.pos,batch.batch)
        return {"energy":output.squeeze(-1)}