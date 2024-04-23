import torch

def mse(y_true, y_pred):
    mse_fuc = torch.nn.MSELoss()
    return mse_fuc(y_true,y_pred)

def mae(y_true, y_pred):
    mae_fuc = torch.nn.L1Loss()
    return mae_fuc(y_true,y_pred)

class LossCalculator:
    def __init__(self,
                 config,
                 ):
        self.losses = {}
        self.loss_fuction = config["loss_fuction"]
        self.real_loss_need = config["real_loss_need"]
        self.loss_weight = config["loss_weight"]
        for target,loss_name in self.loss_fuction.items():
            try:
                self.losses[f"{target}_{loss_name}"] = globals()[loss_name]
            except KeyError:
                raise ValueError(f"Loss function '{loss_name}' is not defined.")
        if self.real_loss_need:
           self.real_loss_fuc = mae
           self.mean = {}
           self.std = {}
           for target in self.losses.keys():
               name  = target.split("_")[0]
               self.mean[name] = 0.0
               self.std[name] = 1.0
    def calculate(self,
                  y_pred, 
                  y_true,
                  mean = None,
                  std = None,
                  mode = "train"):
        
        if mean == None and std == None:
            mean = self.mean
            std = self.std
        
        y_pred_norm = {}
        for target,value in y_pred.items():
            y_pred_norm[target] = value*std[target]+mean[target]
        y_true_norm = {}
        for target,value in y_true.items():
            y_true_norm[target] = (value - mean[target])/std[target] 
        
        loss_values = {}
        for loss_name, func in self.losses.items():
            target= loss_name.split("_")[0]
            if mode in ["train", "val", "test"]:  
                loss_values[f"{mode}_{loss_name}"] = func(y_true_norm[target], y_pred[target])
        if self.loss_weight:
            loss = 0
            ## 补充
            for loss_name,value in loss_values.items():
                target = loss_name.split("_")[1]
                loss += value*self.loss_weight[target]
        
        if self.real_loss_need:
            for target in y_true.keys():
                loss_values[f"{mode}_{target}_real"] = self.real_loss_fuc(y_true[target],y_pred_norm[target])
        
        loss_values["loss"] = loss
        return loss_values