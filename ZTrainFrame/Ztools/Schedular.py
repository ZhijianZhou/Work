import torch
from torch.optim.lr_scheduler import _LRScheduler,ReduceLROnPlateau,ExponentialLR,LinearLR
class ZScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer, 
                 warmup_steps, 
                 max_lr, 
                 base_lr=1e-7,
                 strategy="Linear",
                 strategy_param=None, 
                 last_step= -1,
                 last_epoch = -1):
        ## warmup strategy
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.last_epoch = -1
        self.last_step = -1
        ## schedular stratgey
        if strategy == "Linear":
            self.after_warmup_scheduler = LinearLR(optimizer,
                                                   start_factor=strategy_param["start_factor"],
                                                   end_factor=strategy_param["end_factor"],
                                                   total_iters=strategy_param["total_iters"])
        elif strategy == "Exponnet":
            self.after_warmup_scheduler = ExponentialLR(optimizer, 
                                                        gamma=strategy_param["gama"])
        self.reduce_on_plateau_scheduler = ReduceLROnPlateau(optimizer,
                                                             mode=strategy_param["mode"],
                                                             factor=strategy_param["factor"], 
                                                             patience=strategy_param["patience"], 
                                                             verbose=True, 
                                                             min_lr=1e-7)
        super(ZScheduler, self).__init__(optimizer, last_epoch,last_step)
    
    def get_lr(self):
        if self.last_step ==None:
            self.last_step = -1
        if self.last_step < self.warmup_steps:
            # Warm up phase
            lr = ((self.max_lr - self.base_lr) / self.warmup_steps) * self.last_step + self.base_lr
            return [lr for _ in self.optimizer.param_groups]
        else:
            # After warmup, we delegate to the after warmup scheduler
            return self.after_warmup_scheduler.get_last_lr()
    
    def step(self, metrics=None, epoch=None,step=None):
        if self.last_step ==None:
            self.last_step = -1
        if self.last_step <= self.warmup_steps:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
            self.last_step = step
        elif self.last_step > self.warmup_steps and epoch == None:
            pass
        elif self.last_step > self.warmup_steps and epoch != None:
            self.last_epoch = epoch
            self.after_warmup_scheduler.step()
            if metrics is not None:
                self.reduce_on_plateau_scheduler.step(metrics)