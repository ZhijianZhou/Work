import yaml
from . import utils
import importlib

class GraphDataset:
    def __init__(self,
                 config,
                 model_name,
                 ):
        self.config = config
        if config["config_path"] :
            self.dataset_config = utils.read_config(config["config_path"])
        else :
            self.dataset_config = False
        self.dataset_name = config["name"]
        self.model_name = model_name
        self.Dataset = self.dataset_register()
    def dataset_register(self):
        module_name = f"ZTrainFrame.ModelZoo.{self.model_name}.dataset"
        try:
            dataset_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")
            return None

        try:
            if hasattr(dataset_module, self.dataset_name):
                dataset_class = getattr(dataset_module, self.dataset_name)
                return dataset_class
            else:
                print(f"Dataset {self.dataset_name} not found in module {module_name}.")
                
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return None

    def get_dataset(self,
                    dataset_type,
                    transform = None,
                    pre_transform = None,
                    pre_filter = None):
        assert dataset_type in ['train', 'val', 'test'], "dataset_type must be one of ['train', 'val', 'test']"
        dataset = self.Dataset(model_name = self.model_name,
                               config = self.config,
                               dataset_config =self.dataset_config,
                               dataset_type = dataset_type,
                               transform = transform,
                               pre_transform = pre_transform,
                               pre_filter = pre_filter,
                               ) 
        return dataset