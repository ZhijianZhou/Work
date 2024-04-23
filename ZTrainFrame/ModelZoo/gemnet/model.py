from ZTrainFrame.ModelZoo.gemnet.gemnet_oc import GemNetOC
from torch import nn
class gemnet(nn.Module):
    def __init__(self,
                 config, 
                 ):
        super().__init__()

        self.model = GemNetOC(
            regress_forces = config["regress_forces"],
            otf_graph = config["otf_graph"],
            num_spherical = config["num_spherical"],
            num_radial = config["num_radial"],
            num_blocks = config["num_blocks"],
            emb_size_atom = config["emb_size_atom"],
            emb_size_edge= config["emb_size_edge"],
            emb_size_trip_in= config["emb_size_trip_in"],
            emb_size_trip_out= config["emb_size_trip_out"],
            emb_size_quad_in= config["emb_size_quad_in"],
            emb_size_quad_out= config["emb_size_quad_out"],
            emb_size_aint_in=config["emb_size_aint_in"],
            emb_size_aint_out=config["emb_size_aint_out"],
            emb_size_rbf =config["emb_size_rbf"],
            emb_size_cbf=config["emb_size_cbf"],
            emb_size_sbf=config["emb_size_sbf"],
            num_before_skip=config["num_before_skip"],
            num_after_skip=config["num_after_skip"],
            num_concat=config["num_concat"],
            num_atom=config["num_atom"],
            num_output_afteratom=config["num_output_afteratom"],
            cutoff=config["cutoff"],
            cutoff_qint=config["cutoff_qint"],
            cutoff_aeaint=config["cutoff_aeaint"],
            cutoff_aint=config["cutoff_aint"],
            num_targets=config["num_targets"]
        )
    def forward(self, batch):

        output = self.model(batch)
        return {"energy":output["energy"].squeeze(-1)}