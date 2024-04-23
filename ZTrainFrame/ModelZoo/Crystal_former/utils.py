import torch
from torch.utils.tensorboard import SummaryWriter
import os
import logging

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def learning_rate_with_warmup(current_step, warmup_steps, peak_lr):
    """
    根据当前训练步数计算应用 warmup 策略后的学习率。

    Args:
    current_step: 当前训练步数。
    warmup_steps: Warmup 阶段的步数。
    peak_lr: Warmup 结束后的学习率。

    Returns:
    当前步数对应的学习率。
    """
    if current_step < warmup_steps:
        return peak_lr * current_step / warmup_steps
    else:
        return peak_lr
    
class TensorBoardWriter:
    def __init__(self, log_dir,print_dir, accumulation_steps=10):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(print_dir):
            os.makedirs(print_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_data = {}
        logging.basicConfig(filename=os.path.join(print_dir,"log.log"), level=logging.INFO)

    def write_dict(self, data_dict, step,mode = "train"):
        for key, value in data_dict.items():
            if key not in self.accumulated_data:
                self.accumulated_data[key] = []
            self.accumulated_data[key].append(value)

        
        if mode == "val":
            self._write_data(data_dict, step)
            self._print_log(data_dict, step)
            self.accumulated_data = {}
        else:
            self.current_step += 1
            if self.current_step % self.accumulation_steps == 0 or self.current_step<10 :
                averaged_data = {key: sum(values) / len(values) for key, values in self.accumulated_data.items()}
                self._write_data(averaged_data, step)
                self.accumulated_data = {}
                self._print_log(averaged_data, step)

    def _write_data(self, data_dict, step):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(key, value.item(), step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            else:
                raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")

    def _print_log(self, data_dict, step):
        log_str = f"Step {step} | "
        for key, value in data_dict.items():
            log_str += f"{key}: {value} | "
        logging.info(log_str)
    def print_config(self,data_dict):
        log_str=""
        for key, value in data_dict.items():
            log_str += f"{key}: {value} | \n"
        logging.info(log_str)

    def close(self):
        self.writer.close()