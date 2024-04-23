from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch_geometric.loader import DataLoader
import os


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

        atomrefs =  [
            -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
            -2713.48485589
        ]
        out = torch.zeros(100)
        out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs)
        atomref = out
        self.atom_ref = atomref
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
            energy = molecular["energy"] - np.sum([self.atom_ref[i] for i in elements])
            if self.force :
                mol["force"] = molecular["force"]
            data_mol = Data(atomic_numbers=torch.tensor(elements,dtype=torch.long),
                            natoms = torch.tensor(len(elements),dtype = torch.long),
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
        print("calculating mean and std")
        mean = {}
        std = {}

        # 假设self是一个列表形式的数据集
        data_length = len(self)
        
        # 对每个目标属性tgt进行操作
        for tgt in self.target:
            # 预先为当前属性分配空间
            ys = np.zeros(data_length)
            
            # 一次性填充ys数组
            for i, data in enumerate(self):
                value = getattr(data, tgt, None)
                ys[i] = value.item() if value is not None else np.nan  # np.nan处理缺失值
            
            # 使用numpy直接计算
            mean[tgt] = np.nanmean(ys)  # 忽略nan值
            std[tgt] = np.nanstd(ys)
        
        print("calculate mean std over")
        # print("calculating mean and std")
        # mean = {}
        # std = {}
        # for tgt in self.target:
        #     ys = np.array([getattr(data, tgt, None).item() for data in self])
        #     mean[tgt] = np.mean(ys)
        #     std[tgt] = np.std(ys)
        # print("caculate mean std over")
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
            data_mol = Data(atomic_numbers=torch.tensor(elements,dtype=torch.long),
                            natoms = torch.tensor(len(elements),dtype = torch.long),
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

              