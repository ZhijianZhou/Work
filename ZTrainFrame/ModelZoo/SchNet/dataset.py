from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch_geometric.loader import DataLoader
import os
import tqdm

class QMDataset(InMemoryDataset):
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

        self.force = config["force"]

        super(QMDataset, self).__init__(config["root"], transform, pre_transform, pre_filter)
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
        print("calculating mean and std")
        mean = {}
        std = {}
        for tgt in self.target:
            ys = np.array([getattr(data, tgt, None).item() for data in self])
            mean[tgt] = np.mean(ys)
            std[tgt] = np.std(ys)
        print("caculate mean std over")
        return mean, std

    def process(self):
        data = np.load(self.raw_file_names,allow_pickle=True)
        data_list = []

        for molecular in tqdm(data):
            mol = {}
            mol_name = molecular["mol_name"]
            elements = molecular["elements"]
            pos = molecular["position"]
            energy = molecular["energy"]
            if self.force :
                mol["force"] = molecular["force"]
            data_mol = Data(x=torch.tensor(elements,dtype=torch.long),
                             pos=torch.tensor(pos,dtype=torch.float),
                             energy=torch.tensor(energy,dtype=torch.float),
                             mol_name = torch.tensor(np.array([ord(i)for i in mol_name]),dtype=torch.long))
            data_list.append(data_mol)
        

        processed_data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((processed_data, slices), self.processed_paths[0])
    def __getitem__(self, idx):
        # 根据索引获取数据项
        data = super(QMDataset, self).__getitem__(idx)
            
        return data



class ZCDataset(InMemoryDataset):
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

        self.force = config["force"]

        super(ZCDataset, self).__init__(config["root"], transform, pre_transform, pre_filter)
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
                for i, data in enumerate(tqdm.tqdm(self)):
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

    def process(self):
        data = np.load(self.raw_file_names,allow_pickle=True)
        data_list = []

        for molecular in tqdm(data):
            mol = {}
            mol_name = molecular["mol_name"]
            elements = molecular["elements"]
            pos = molecular["coordinates"]
            energy = molecular["energy"]
            if self.force :
                mol["force"] = molecular["force"]
            data_mol = Data(x=torch.tensor(elements,dtype=torch.long),
                             pos=torch.tensor(pos,dtype=torch.float),
                             energy=torch.tensor(energy,dtype=torch.float),
                             mol_name = torch.tensor(np.array([ord(i)for i in mol_name]),dtype=torch.long))
            data_list.append(data_mol)
        

        processed_data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((processed_data, slices), self.processed_paths[0])
    def __getitem__(self, idx):
        # 根据索引获取数据项
        data = super(ZCDataset, self).__getitem__(idx)
            
        return data

              