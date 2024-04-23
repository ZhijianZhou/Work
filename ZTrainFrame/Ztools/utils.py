import yaml
import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from collections import defaultdict

def read_config(config_path):
    """
    Reads a configuration file and returns its contents as a dictionary.
    Supports YAML and JSON formats.

    Args:
    - config_path (str): The file path to the configuration file.

    Returns:
    - dict: The configuration as a dictionary.

    Raises:
    - ValueError: If the file format is not supported.
    """
    _, file_extension = os.path.splitext(config_path)
    # Normalize the file extension to ensure compatibility
    file_extension = file_extension.lower()

    if file_extension == ".yaml" or file_extension == ".yml":
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    elif file_extension == ".json":
        with open(config_path, "r") as file:
            config = json.load(file)
    else:
        raise ValueError("Unsupported configuration file format. Only JSON and YAML are supported.")

    return config

class TensorBoardWriter:
    def __init__(self, log_dir, print_dir, accumulation_steps=10):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_data = defaultdict(list)
        logging.basicConfig(filename=os.path.join(print_dir, "log.log"), level=logging.INFO)

    def write_dict(self, data_dict, step, mode="train"):
        if mode == "val":
            self._write_data(data_dict, step)
            self._print_log(data_dict, step)
            self.accumulated_data.clear()
        else:
            for key, value in data_dict.items():
                self.accumulated_data[key].append(value)
            self.current_step += 1
            if self.current_step % self.accumulation_steps == 0 or self.current_step < 10:
                averaged_data = {key: sum(values) / len(values) for key, values in self.accumulated_data.items()}
                self._write_data(averaged_data, step)
                self._print_log(averaged_data, step)
                self.accumulated_data.clear()

    def _write_data(self, data_dict, step):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(key, value.item(), step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            else:
                raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")

    def _print_log(self, data_dict, step):
        log_str = f"Step {step} | " + " | ".join(f"{key}: {value}" for key, value in data_dict.items())
        logging.info(log_str)

    def print_config(self, data_dict):
        log_str = "\n".join(f"{key}: {value} |" for key, value in data_dict.items())
        logging.info(log_str)

    def close(self):
        self.writer.close()
