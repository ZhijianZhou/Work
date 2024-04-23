
import os
import json
import numpy as np
from tqdm import tqdm
import copy
import random
import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, InMemoryDataset, Batch
from jarvis.core.specie import chem_data, get_node_attributes
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata

import ZTrainFrame

def picturetocoords(lattice,picture,coords):  
    return np.array(picture)@np.array(lattice)+np.array(coords)
class CrystalDataset(InMemoryDataset):
    def __init__(
                 self,
                 model_name,
                 config,
                 dataset_type,
                 dataset_config,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        self.root = config["root"]
        self.data_path = dataset_type+".npy"
        self.name = dataset_type
        self.processdir = model_name+"_"+config["processdir"]
        self.target = config["target"]

        self.dictionary = dataset_config["dictionary"]
        self.ispretrain = config["pretrain"]
        self.inf_feature = config["inf_feature"]
        self.self_edge_feature = config["self_edge_feature"]

        super(CrystalDataset, self).__init__(config["root"], transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return os.path.join(self.root, self.data_path)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processdir)

    @property
    def processed_file_names(self):
        return self.name + '.pt'
    
    def calc_stats(self):
        mean_std_dict_name = os.path.join(self.processed_dir,self.name+".npy")
        if os.path.exists(mean_std_dict_name):
            mean_std = np.load(mean_std_dict_name,allow_pickle=True)
            mean = mean_std.item()["mean"]
            std = mean_std.item()["std"]
        else:
            print("calculating mean and std")
            mean = {}
            std = {}
            data_length = len(self)
            for tgt in self.target:
                ys = np.zeros(data_length)
                for i, data in enumerate(tqdm(self)):
                    value = getattr(data, tgt, None)
                    ys[i] = value.item() if value is not None else np.nan  # np.nan处理缺失值
                mean[tgt] = np.nanmean(ys)  
                std[tgt] = np.nanstd(ys)
            print("calculate mean std over")
            mean_std = {}
            mean_std["mean"] = mean
            mean_std["std"] = std
            np.save(mean_std_dict_name,mean_std)
        return mean, std

    def process_dft3d_data(self):
        crystal_data = []
        infinite_funcs =  ["zeta", "zeta", "exp"]
        infinite_params = [0.5, 3.0, 3.0]
        with open("/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Data/dft_3d/jdft_3d-12-12-2022.json","r") as fp:
            dft_3d = json.load(fp)
        with open("/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Data/dft_3d/dft_3d_formation_energy_peratom.json","r") as fp:
            split_index = json.load(fp)
        for mat in tqdm.tqdm(dft_3d):
            count = 0
            pos = []
            period_index = []
            edge_index_i = []
            edge_index_j = []
            atom = {}
            crystal = {}
            crystal["jid"] = copy.copy(mat['jid'])
            crystal["coords"] = copy.copy(mat["atoms"]['coords'])
            crystal["elements"] = copy.copy(mat["atoms"]['elements'])
            crystal["lattice_mat"] = copy.copy(mat["atoms"]["lattice_mat"])
            crystal["formation_energy_peratom"] = copy.copy(mat["formation_energy_peratom"])
            elements_count = len(crystal["elements"])
            structure = Atoms.from_dict(mat['atoms'])
            lattice_mat = np.array(crystal["lattice_mat"])
            coords = np.array(crystal["coords"])
            N = structure.get_all_neighbors(r=5.0)
            ## calculate real index for calculate angle
            for id,i in enumerate(N):
                soure_name = str(id)+"_"+str(0)+"_"+str(0)+"_"+str(0)
                if soure_name not in atom.keys():
                        atom[soure_name] = {}
                        atom[soure_name]["real_index"] = count
                        atom[soure_name]["period_index"] = id
                        pic = [0.0,0.0,0.0]
                        atom[soure_name]["picture"] = pic
                        atom[soure_name]["pos"] = picturetocoords(lattice_mat,pic,coords[id])
                        pos.append(atom[soure_name]["pos"])
                        period_index.append(id)
                        count+=1
                for j in i:
                    source,target,r,(pic_x,pic_y,pic_z) = j
                    target_name = str(target)+"_"+str(int(pic_x))+"_"+str(int(pic_y))+"_"+str(int(pic_z))
                    if target_name not in atom.keys():
                        atom[target_name] = {}
                        atom[target_name]["real_index"] = count
                        atom[target_name]["period_index"] = target
                        pic = [pic_x,pic_y,pic_z]
                        atom[target_name]["picture"] = pic
                        atom[target_name]["pos"] = picturetocoords(lattice_mat,pic,coords[target])
                        pos.append(atom[target_name]["pos"])
                        period_index.append(target)
                        count+=1
                    edge_index_i.append(atom[soure_name]["real_index"])
                    edge_index_j.append(atom[target_name]["real_index"])
            u = torch.arange(0, elements_count , 1).unsqueeze(1).repeat((1, elements_count )).flatten().long()
            v = torch.arange(0,elements_count , 1).unsqueeze(0).repeat((elements_count , 1)).flatten().long()
            inf_edge_index = torch.stack([u, v])
            lattice_mat = structure.lattice_mat.astype(dtype=np.double)
            vecs = structure.cart_coords[u.flatten().numpy().astype(int)] - structure.cart_coords[
                v.flatten().numpy().astype(int)]

            inf_edge_attr = torch.FloatTensor(np.stack([getattr(ZTrainFrame.ModelZoo.Crystal_former.algorithm, func)(vecs, lattice_mat, param=param, R=5)
                                            for func, param in zip(infinite_funcs, infinite_params)], 1))
            edges = nearest_neighbor_edges(atoms=structure, cutoff=5.0, max_neighbors=16)
            u0, v0, r = build_undirected_edgedata(atoms=structure, edges=edges)
            crystal["edge_index"] = [u0,v0]
            crystal["edge_distance"] = r.norm(dim=-1)
            crystal["inf_edge_attr"] = inf_edge_attr
            crystal["inf_edge_index"] = inf_edge_index

            crystal["real_edge_index"] = [edge_index_i,edge_index_j]
            crystal["period_index"] = period_index
            crystal["real_pos"] = pos
            crystal_data.append(crystal)
        train = []
        test = []
        val = []
        for i in crystal_data:
            if i["jid"] in split_index["test"].keys():
                test.append(i)
            elif i["jid"] in split_index["val"].keys():
                    val.append(i)
            else:
                    train.append(i)
        np.save("/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Data/dft_3d/train.npy",train)
        np.save("/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Data/dft_3d/val.npy",val)
        np.save("/cpfs01/projects-HDD/cfff-282dafecea22_HDD/zhouzhijian/Work/Data/dft_3d/test.npy",test)
    def process(self):
        mat_data = np.load(self.raw_file_names,allow_pickle=True)

        data_list = []
        
       
        for mat in tqdm(mat_data):
            jid = mat["jid"]
            elements = [get_node_attributes(i,"atomic_number")[0] for i in mat["elements"]]
            elements_count = torch.tensor(len(elements),dtype=torch.long)
            formation_energy_peratom = torch.tensor(mat["formation_energy_peratom"],dtype=torch.float32)
            
            u,v = mat["edge_index"]
            u = torch.tensor(u,dtype=torch.long)
            v = torch.tensor(v,dtype=torch.long)
            edge_distance = torch.tensor(mat["edge_distance"],dtype=torch.float32)
            
            if self.inf_feature and self.self_edge_feature == False:
                inf_edge_attr = torch.tensor(mat["inf_edge_attr"],dtype=torch.float32)
                inf_edge_index = torch.tensor(mat["inf_edge_index"],dtype=torch.long)
                data = Data(x=torch.tensor(elements,dtype=torch.long), 
                            edge_index=torch.stack([u, v]),
                            edge_attr=edge_distance,
                            FormationEnergyPeratom=formation_energy_peratom,
                            elements_count=elements_count,
                            jid = jid,
                            inf_edge_attr = inf_edge_attr,
                            inf_edge_index = inf_edge_index)
            elif self.self_edge_feature and self.inf_feature:
                inf_edge_attr =  torch.tensor(mat["inf_edge_attr"],dtype=torch.float32)
                inf_edge_index =  torch.tensor(mat["inf_edge_index"],dtype=torch.long)
                self_edge_index = torch.tensor(torch.stack(mat["self_edge_index"]),dtype=torch.long)
                self_edge_attr =  torch.tensor(mat["self_edge_attr"],dtype=torch.float32)
                data = Data(x=torch.tensor(elements,dtype=torch.long), 
                            edge_index=torch.stack([u, v]),
                            edge_attr=edge_distance,
                            FormationEnergyPeratom=formation_energy_peratom,
                            elements_count=elements_count,
                            jid = jid,
                            inf_edge_attr = inf_edge_attr,
                            inf_edge_index = inf_edge_index,
                            self_edge_attr = self_edge_attr,
                            self_edge_index = self_edge_index)
            else:
                data = Data(x=torch.tensor(elements,dtype=torch.long), 
                            edge_index=torch.stack([u, v]),
                            edge_attr=edge_distance,
                            FormationEnergyPeratom=formation_energy_peratom,
                            elements_count=elements_count,
                            jid = jid)
    
            data_list.append(data)
        


        mat_data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((mat_data, slices), self.processed_paths[0])
    def __getitem__(self, idx):
        # 根据索引获取数据项
        data = super(CrystalDataset, self).__getitem__(idx)  
        return data



class AddFeatureMask(BaseTransform):
    def __init__(self, config,
                 mask_rate=0.15):
        # mask_rate定义了需要被mask的节点特征的比例
        self.config = config
        self.mask_rate = mask_rate
    def select_atoms(self,atom_count):
        """
        Choose 15% atoms to predict.

        Args:
            atom_count (int): the number of molecular's atoms.

        Returns:
            select_indice : np.array
            mask: np.array 
        """
        num_points_to_select = int(self.mask_rate * int(atom_count))
        select_indices = torch.randperm(int(atom_count))[:num_points_to_select]
        mask = torch.zeros(atom_count)
        mask[select_indices] = 1.0

        return select_indices,mask
        
    def masked_atom_prediction(self,elements, selected_indices):
        """
        mask atoms .

        Args:
            elements (list) : the atoms of molecular.
            selected_indices (ndarry) : the selected atoms to mask
            dictionary (dict) : the dictionary of atoms

        Returns:
            masked_elements (list) : replaced the sequence of atoms
        """

        masked_elements = elements
        # With 80% probability, replace the atom with "[MASK]"
        mask_indices = selected_indices[torch.rand(len(selected_indices)) < 0.8]
        masked_elements[mask_indices] = 93

        # With 10% probability, replace the atom with a random atom (excluding "[PAD]" and "[MASK]")
        # Assuming dictionary indices are continuous and start from 0 to len(dictionary)-1
        random_indices = selected_indices[torch.rand(len(selected_indices)) >= 0.8]
        if len(random_indices) > 0:
            random_elements = torch.randint(low=1, high=92, size=(len(random_indices),))
            masked_elements[random_indices] = random_elements

        return masked_elements
    def __call__(self, data):
        # data = copy.deepcopy(data)
        selected_indice,atoms_mask = self.select_atoms(data.x.shape[0])
        
        
        ## mask atom
        masked_elements = self.masked_atom_prediction(data.x.clone(),selected_indice)
        
        data.elements_mask = atoms_mask.to(torch.bool)
        data.elements = data.x.clone()
        data.x = masked_elements.clone()
        return data