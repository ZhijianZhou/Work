import os
import time
import random
import pandas as pd
import csv
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from ZTrainFrame.Ztools import utils
from ZTrainFrame.Ztools import dataset_loader
from ZTrainFrame.Ztools import model_loader
from ZTrainFrame.Ztools import loss_function
from ZTrainFrame.Ztools import Schedular

class Ztrainer:
    def __init__(self, config_path,local_rank,check_point,description = "None"):
        ## Load config
        # if check_point:
        #     self.load_checkpoint(check_point,load_config=True)
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
        self.model = self.model_fuc.get_model()
        ## Set the optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["model"]["learning_rate"])
        self.scheduler = Schedular.ZScheduler(self.optimizer,warmup_steps=self.config["schedular"]["warmup_steps"],
                                              max_lr=self.config["schedular"]["max_lr"],
                                              base_lr=self.config["schedular"]["base_lr"],
                                              strategy=self.config["schedular"]["strategy"],
                                              strategy_param=self.config["schedular"]["strategy_param"],
                                              )
        ## Set training process parameters
        self.step = 0
        self.start_epoch = 0
        self.epoch_num = self.config["model"]["epoch_num"]
        self.val_freq = self.config["model"]["val_freq"]
        if check_point :
            print("load check point")
            self.load_checkpoint(file_path=check_point)
        ## log
        self.description = description
        self.best_metrics = 1000

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
        test_dataset = self.dataset.get_dataset(dataset_type="test")
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset )
        test_sampler =DistributedSampler(test_dataset)
        self.train_loader = DataLoader(train_dataset,sampler = train_sampler, batch_size=self.config["model"]["batch_size"])
        self.val_loader = DataLoader(val_dataset,sampler = val_sampler, batch_size=self.config["model"]["batch_size"])
        self.test_loader = DataLoader(test_dataset,sampler = test_sampler, batch_size=self.config["model"]["batch_size"])
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
            
        for epoch in tqdm.tqdm(range(self.start_epoch,self.epoch_num+1)):
            self.epoch = epoch
            self.train_epoch()
            if epoch % self.val_freq == 0 or epoch == 0:
                val_loss = self.validate_epoch()
                test_loss = self.test_epoch()
                self.scheduler.step(metrics=val_loss,
                                    epoch=epoch)
                if self.local_rank == 0:
                    if val_loss < self.best_metrics:
                        self.best_metrics = val_loss
                        ifbest = True
                    else:
                        ifbest = False
                    self.save_checkpoint(epoch = epoch+1,ifval=True,ifbest=ifbest)
            else:    
                self.scheduler.step(epoch=epoch)
                if self.local_rank == 0:
                    self.save_checkpoint(epoch = epoch+1,ifval=False)
    def run(self):
        self.init_helper()
        print("-------------Loading dataset-------------")
        train_dataset = self.dataset.get_dataset(dataset_type="train")
        val_dataset = self.dataset.get_dataset(dataset_type="val")
        test_dataset = self.dataset.get_dataset(dataset_type="test")
        self.train_loader = DataLoader(train_dataset, batch_size=self.config["model"]["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config["model"]["batch_size"], shuffle=False)
        self.test_loader = DataLoader(test_dataset,batch_size=self.config["model"]["batch_size"], shuffle=False)
        if self.norm_need:
            self.mean,self.std = self.train_loader.dataset.calc_stats()
        else:
            self.mean,self.std = None,None
        print("mean and std:","\n",self.mean,"\n",self.std)
        self.model.to(self.device)
        print("-------------Start training-------------")
        for epoch in tqdm.tqdm(range(self.start_epoch,self.epoch_num+1)):
            self.epoch = epoch
            self.train_epoch()
            if epoch % self.val_freq == 0 or epoch == 0:
                val_loss = self.validate_epoch()
                if val_loss < self.best_metrics:
                    self.best_metrics = val_loss
                    ifbest = True
                else:
                    ifbest = False
                self.scheduler.step(metrics=val_loss,
                                    epoch=epoch)
                test_loss = self.test_epoch(ddp=False)
                self.save_checkpoint(epoch=epoch+1,mode="test",ifval=True,ifbest=ifbest)
            else:    
                self.scheduler.step(epoch=epoch)
                self.save_checkpoint(epoch=epoch+1,mode="test",ifval=False)
    
    def inference(self,model_pth):
        # self.load_model(model_pth)
        
        test_dataset = self.dataset.get_dataset(dataset_type="test")
        self.test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        if self.norm_need:
            train_dataset = self.dataset.get_dataset(dataset_type="train")
            self.train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
            self.mean,self.std = self.train_loader.dataset.calc_stats()
            print(self.test_loader.dataset.calc_stats())
            print(self.mean,self.std)
        else:
            self.mean,self.std = None,None
        self.model.to(self.device)
        self.model.eval()
        iterations = tqdm.tqdm(self.test_loader,leave=True,unit="iteration")
        self.accumulated_data = {}
        with torch.no_grad():
            results = {}
            for tgt in self.target:
                results[tgt] = []
            for batch in iterations:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                targets ={}
                
                for tgt in self.target:
                    targets[tgt] = getattr(batch,tgt)
                    results[tgt] = results[tgt] + targets[tgt].cpu().tolist()
                loss_result = self.loss_fuc.calculate(y_pred=outputs,
                                                y_true=targets,
                                                mean=self.mean,
                                                std=self.std,
                                                mode="test"
                                                )
                for key, value in loss_result.items():
                    if key not in self.accumulated_data:
                        self.accumulated_data[key] = []
                    self.accumulated_data[key].append(value)
                iterations.set_postfix(real_loss=f'{loss_result["loss"].item():.4f}')
        averaged_data = {key: sum(values) / len(values) for key, values in self.accumulated_data.items()}
        for i,j in averaged_data.items():
            print(i,j)
        with open('my_dict.pkl', 'wb') as f:
            pickle.dump(results, f)
    def init_helper(self):
        self.exp_father = os.path.join(self.config["model"]["exp_path"],self.config["model"]["dataset"],self.config["model"]["name"],self.config["model"]["task"])
        if os.path.exists(self.exp_father)  == False:
            os.makedirs(self.exp_father)
        version = time.time()
        self.exp = os.path.join(self.exp_father,"version"+str(version))
        os.makedirs(self.exp)

        self.log_father = os.path.join(self.config["model"]["log_path"],self.config["model"]["dataset"],self.config["model"]["name"],self.config["model"]["task"])
        if os.path.exists(self.log_father)  == False:
            os.makedirs(self.log_father)
        
        self.log_path = os.path.join(self.log_father,"version"+str(version))
        os.makedirs(self.log_path)
        self.writer = utils.TensorBoardWriter(log_dir=self.log_path,print_dir=self.exp,accumulation_steps=self.config["model"]["print_steps"])
        
        csv_filename = os.path.join(self.config["model"]["exp_path"],self.config["model"]["dataset"],self.config["model"]["name"],self.config["model"]["task"],"experiment_log.csv")
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['time', 'description', 'result']  # 替换为你的列名
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 如果文件不存在，则写入列名
            if not os.path.isfile(csv_filename):
                writer.writeheader()
            csv_data = {'time':str(version),'description':str(self.description),'result':str(0)}
            writer.writerow(csv_data)
        self.writer.print_config(self.config)
        

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
            
            self.step += 1
            self.scheduler.step(step=self.step)
            iterations.set_postfix(loss=f'{loss.item():.4f}')
            loss_result["lr"] = self.scheduler.get_lr()[0]
            loss_result["epoch"] = self.epoch
            if self.local_rank == 0:
                self.writer.write_dict(loss_result,self.step,"train")
        
    def validate_epoch(self):
        self.model.eval()
        iterations = tqdm.tqdm(self.val_loader,leave=True,unit="iteration")
        self.accumulated_data = {}
        torch.cuda.empty_cache()
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
    def test_epoch(self,ddp=True):
        self.model.to(self.device)
        if self.norm_need:
            self.mean,self.std = self.train_loader.dataset.calc_stats()
        else:
            self.mean,self.std = None,None
        test_dataset = self.dataset.get_dataset(dataset_type="test")
        self.test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        self.model.eval()
        iterations = tqdm.tqdm(self.test_loader,leave=True,unit="iteration")
        self.accumulated_data = {}
        torch.cuda.empty_cache()
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
                                                mode="test"
                                                )
                for key, value in loss_result.items():
                    if key not in self.accumulated_data:
                        self.accumulated_data[key] = []
                    self.accumulated_data[key].append(value)
                iterations.set_postfix(real_loss=f'{loss_result["loss"].item():.4f}')
        self.averaged_data = {key: sum(values) / len(values) for key, values in self.accumulated_data.items()}
        if ddp == False:
            self.writer.write_dict(self.averaged_data,self.step,"val")
        if ddp:
            for key in self.averaged_data.keys():
                value_tensor = torch.tensor(self.averaged_data[key], device=self.device)
                dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)  # Sum values from all GPUs
                if dist.get_rank() == 0:  # Optionally, do the following only on the master process
                    self.averaged_data[key] = value_tensor.item() / dist.get_world_size()  # Average the result

            # Only on rank 0 if you want to print or log results
            if dist.get_rank() == 0:
                averaged_data = {key: self.averaged_data[key] for key in self.averaged_data}
                self.writer.write_dict(averaged_data,self.step,"val")
                return averaged_data["loss"]
            else:
                return 0
        else:
            return 0

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

    def save_checkpoint(self,epoch,mode="ddp",ifval=True,ifbest=False):
        if mode == "ddp":
            model_state_dict = self.model.module.state_dict()
            
        else:
            model_state_dict = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'mean':self.mean,
            'std':self.std,
            'step': self.step,
            'config':self.config
        }
        file_path_latest=os.path.join(self.exp,f'checkpoint_leatest.pt')
        torch.save(checkpoint, file_path_latest)
        if ifbest:
            file_path=os.path.join(self.exp,f'checkpoint_best.pt')
            torch.save(checkpoint, file_path)
            torch.save(model_state_dict, os.path.join(self.exp,"best_model.pkl"))
            print(f"Best Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path='checkpoint.pt',load_config = False):
        if os.path.isfile(file_path):
            if load_config:
                checkpoint = torch.load(file_path)
                self.config = checkpoint["config"]
        
            else:
                checkpoint = torch.load(file_path)
                self.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.step = checkpoint['step']
                if "config" in checkpoint.keys():
                    self.config = checkpoint["config"]
                print(f"epoch{self.start_epoch}")
                # self.mean=checkpoint['mean']
                # self.std = checkpoint['std']
                print(f"Checkpoint loaded from {file_path}")
        else:
            print("No checkpoint found at specified path.")
    def remove_prefixes(self,state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', 'model.')  # Adjust the prefix according to actual requirement
            new_state_dict[new_key] = value
        return new_state_dict

    def load_model_state_dict(self,model_pth):
        checkpoint = torch.load(model_pth)
        self.model.load_state_dict(checkpoint)
    def load_model(self,model_pth):
        print("loading model")
        checkpoint = torch.load(model_pth)
        modified_state_dict = self.remove_prefixes(checkpoint['model_state_dict'])
        self.model.load_state_dict(modified_state_dict)

       