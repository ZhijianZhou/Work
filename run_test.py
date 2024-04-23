# from ZTrainFrame.ModelZoo.eScN.model import escn
from ZTrainFrame.Ztools.trainer import Ztrainer
import argparse
import torch
import os
print(f"{os.getcwd()}")
description = ""
"/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Config/ZC_dimenetpp/config_diment.yaml"
"/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Config/ZC_schnet/config_schnet_ZC.yaml"
"/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Config/dft_3d_Crystal_former/config_crystal_former.yaml"
potnet = "/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Config/dft_3d_potnet/config_potnet.yaml"
trainer = Ztrainer(config_path = potnet,
                   local_rank=0,
                   check_point=False,
                   description="GAT 6" )
trainer.run()
