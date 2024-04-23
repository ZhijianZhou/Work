import os
import random
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from ZTrainFrame.Ztools import utils
from ZTrainFrame.Ztools import dataset_loader
from ZTrainFrame.Ztools import model_loader
from ZTrainFrame.Ztools import loss_function
from ZTrainFrame.Ztools import optimizer

class Ztrainer:
    def __init__(self, config_path,local_rank,check_point):
        ## Load config
        self.config = utils.read_config(config_path)
        ## Initialize the dataset
        self.dataset = dataset_loader.GraphDataset(config = self.config["dataset"],
                                                   model_name = self.config["model"]["name"])
        ## Initialize the Model Function
        self.model_fuc = model_loader.ModelLoader(config=self.config["model"])
        ## Fix the random seed
        self.seed = self.config["model"]["seed"]
        self.set_seed()
        ## Set the loss function
        self.loss_fuc = loss_function.LossCalculator(self.config["loss"])
        # Output whether normalization is needed
        self.norm_need = self.config["loss"]["norm_need"] 
        # Set the prediction target
        self.target = self.config["dataset"]["target"]
        ## Load model
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
#         if local_rank != False:
            
#         else:
#             self.local_rank = 0
#             self.device = torch.device("cuda:"+str(self.config["model"]["cuda_idx"]) if torch.cuda.is_available() else "cpu")
        self.model = self.model_fuc.get_model()
        ## Set the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["model"]["learning_rate"])
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.config["optimizer"]["ex_decay"])
        self.scheduler1 = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5, verbose=True, min_lr=1e-7)
        ## Set training process parameters
        self.step = 0
        self.start_epoch = 0
        self.epoch_num = self.config["model"]["epoch_num"]
        self.val_freq = self.config["model"]["val_freq"]
        self.ex_decay_freq = self.config["optimizer"]["ex_decay_freq"]
        if check_point :
            print("load check point")
            self.load_checkpoint(file_path=check_point)
    def run_ddp(self,rank,nprocs):
        if rank == 0:
            self.init_helper()
        self.local_rank = rank
        # 分布式初始化，对于每个进程来说，都需要进行初始化
        cudnn.benchmark = True
        dist.init_process_group(backend='nccl')
        # 设置进程对应使用的GPU
        torch.cuda.set_device(self.local_rank)
        self.model.cuda(self.local_rank)
        # 使用分布式函数定义模型
        self.model =  torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank ])
        ## dataset
        if self.local_rank == 0:
            print("-------------Loading dataset-------------")
        train_dataset = self.dataset.get_dataset(dataset_type="train")
        val_dataset = self.dataset.get_dataset(dataset_type="val")
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset )
        self.train_loader = DataLoader(train_dataset,sampler = train_sampler, batch_size=self.config["model"]["batch_size"])
        self.val_loader = DataLoader(val_dataset,sampler = val_sampler, batch_size=self.config["model"]["batch_size"])
        if self.norm_need:
            self.mean,self.std = self.train_loader.dataset.calc_stats()
        else:
            self.mean,self.std = None,None
        if self.local_rank == 0:
            if self.norm_need:
                print("Mean and Std:")
                for i,j in zip(self.mean,self.std):
                    print(i,j)
            print("-------------Start training-------------")
        print(self.start_epoch)
        for epoch in tqdm.tqdm(range(self.start_epoch,self.epoch_num+1)):
            print(self.device)
            self.train_epoch()
            if epoch % self.val_freq == 0 or epoch == 0:
                val_loss = self.validate_epoch()
                self.scheduler1.step(val_loss)
                if self.local_rank == 0:
                    self.save_checkpoint(epoch+1)
            if epoch > 0 and epoch % self.ex_decay_freq ==0:
                self.scheduler.step()
    def run(self):
        self.init_helper()
        print("Load train dataset")
        train_dataset = self.dataset.get_dataset(dataset_type="train")
        print("load val dataset")
        val_dataset = self.dataset.get_dataset(dataset_type="val")
        self.train_loader = DataLoader(train_dataset, batch_size=self.config["model"]["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config["model"]["batch_size"], shuffle=False)
        if self.norm_need:
            self.mean,self.std = self.train_loader.dataset.calc_stats()
        else:
            self.mean,self.std = None,None
        print("mean and std:","\n",self.mean,"\n",self.std)
        for epoch in tqdm.tqdm(range(self.start_epoch,self.epoch_num+1)):
            self.train_epoch()
            if epoch % self.val_freq == 0 and epoch == 0:
                val_loss = self.validate_epoch()
                self.scheduler1.step(val_loss)
            if epoch > 0 and epoch % self.ex_decay_freq ==0:
                self.scheduler.step()
    

    def init_helper(self):
        self.exp_father = os.path.join(self.config["model"]["exp_path"],self.config["model"]["dataset"],self.config["model"]["name"],self.config["model"]["task"])
        if os.path.exists(self.exp_father)  == False:
            os.makedirs(self.exp_father)
        version = len(os.listdir(self.exp_father))
        self.exp = os.path.join(self.exp_father,"version"+str(version))
        os.makedirs(self.exp)

        self.log_father = os.path.join(self.config["model"]["log_path"],self.config["model"]["dataset"],self.config["model"]["name"],self.config["model"]["task"])
        if os.path.exists(self.log_father)  == False:
            os.makedirs(self.log_father)
        version = len(os.listdir(self.log_father))
        self.log_path = os.path.join(self.log_father,"version"+str(version))
        os.makedirs(self.log_path)
        self.writer = utils.TensorBoardWriter(log_dir=self.log_path,print_dir=self.exp,accumulation_steps=self.config["model"]["print_steps"])

    def train_epoch(self):
        self.model.train()
        iterations = tqdm.tqdm(self.train_loader,leave=True,unit="iteration")
        for batch in iterations:
            batch = batch.to(self.device)
            outputs = self.model(batch)
            targets ={}
            for tgt in self.target:
                targets[tgt] = getattr(batch,tgt)
            loss_result = self.loss_fuc.calculate(y_pred=outputs,
                                                  y_true=targets,
                                                  mean=self.mean,
                                                  std=self.std,
                                                  mode="train"
                                                  )
            self.optimizer.zero_grad()
            loss = loss_result["loss"]
            loss.backward()
            self.optimizer.step()
            if self.local_rank == 0:
                self.writer.write_dict(loss_result,self.step,"train")
            self.step += 1
            iterations.set_postfix(loss=f'{loss.item():.4f}')
        
    def validate_epoch(self):
        self.model.eval()
       
        iterations = tqdm.tqdm(self.val_loader,leave=True,unit="iteration")
        self.accumulated_data = {}
        with torch.no_grad():
            for batch in iterations:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                targets ={}
                for tgt in self.target:
                    targets[tgt] = getattr(batch,tgt)
                loss_result = self.loss_fuc.calculate(y_pred=outputs,
                                                y_true=targets,
                                                mean=self.mean,
                                                std=self.std,
                                                mode="val"
                                                )
                for key, value in loss_result.items():
                    if key not in self.accumulated_data:
                        self.accumulated_data[key] = []
                    self.accumulated_data[key].append(value)
                iterations.set_postfix(real_loss=f'{loss_result["loss"].item():.4f}')
        averaged_data = {key: sum(values) / len(values) for key, values in self.accumulated_data.items()}
        if self.local_rank == 0:
            self.writer.write_dict(averaged_data,self.step,"val")
        return loss_result["loss"].item()
    
    def test_epoch(self):
        self.test_dataset = self.dataset.get_dataset(dataset_type="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config["model"]["batch_size"], shuffle=False)

        self.model.eval()
        iterations = tqdm.tqdm(self.test_loader,leave=True,unit="iteration")
        self.accumulated_data = {}
        with torch.no_grad():
            for batch in iterations:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                targets ={}
                for tgt in self.target:
                    targets[tgt] = getattr(batch,tgt)
                loss_result = self.loss_fuc.calculate(y_pred=outputs,
                                                y_true=targets,
                                                mean=self.mean,
                                                std=self.std,
                                                mode="train"
                                                )
                for key, value in loss_result.items():
                    if key not in self.accumulated_data:
                        self.accumulated_data[key] = []
                    self.accumulated_data[key].append(value)
                iterations.set_postfix(real_loss=f'{loss_result["loss"].item():.4f}')
        averaged_data = {key: sum(values) / len(values) for key, values in self.accumulated_data.items()}
        
        if self.local_rank == 0:
            self.writer.write_dict(averaged_data,self.step,"val")

    def set_seed(self):
        """
        Set the seed for all sources of randomness to ensure reproducibility.
        :param seed: An integer representing the random seed.
        """
        random.seed(self.seed)  # Python random module.
        np.random.seed(self.seed)  # Numpy module.
        torch.manual_seed(self.seed)  # PyTorch random number generator for CPU.
        
        # 如果使用CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)  # PyTorch random number generator for all GPUs.
            torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.

    def save_checkpoint(self,epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scheduler1_state_dict': self.scheduler1.state_dict(),
            'step': self.step
        }
        file_path=os.path.join(self.exp,f'checkpoint_{epoch}.pt')
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path='checkpoint.pt'):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scheduler1.load_state_dict(checkpoint['scheduler1_state_dict'])
            self.step = checkpoint['step']
            print(f"Checkpoint loaded from {file_path}")
        else:
            print("No checkpoint found at specified path.")