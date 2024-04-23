from torch_geometric.nn.models import DimeNetPlusPlus
from torch import nn
class dimenetpp(nn.Module):
    def __init__(self,
                 config, 
                 ):
        super().__init__()

        self.model = DimeNetPlusPlus (
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            num_blocks=config["num_blocks"],
            int_emb_size=config["int_emb_size"],
            basis_emb_size=config["basis_emb_size"],
            out_emb_channels=config["out_emb_channels"],
            num_spherical=config["num_spherical"],
            num_radial=config["num_radial"],
            cutoff=config["cutoff"],
            max_num_neighbors=config["max_num_neighbors"],
            envelope_exponent=config["envelope_exponent"],
            num_before_skip=config["num_before_skip"],
            num_after_skip=config["num_after_skip"],
            num_output_layers=config["num_output_layers"],
        )
    def forward(self, batch):

        output = self.model(batch.atomic_numbers,batch.pos,batch.batch)
        return {"energy":output.squeeze(-1)}