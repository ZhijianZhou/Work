import importlib
from . import utils  # 确保 utils.read_config 能正常工作

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config["name"]
        self.task = self.config["task"]
        if "config_path" in config:
            self.model_config = utils.read_config(config["config_path"])
        else:
            self.model_config = None
        self.model_fuc = self.load_model()

    def load_model(self):
        """
        动态加载指定模型名的模型。
        """
        module_name = f"ZTrainFrame.ModelZoo.{self.model_name}.model"
        try:
            model_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")
            return None
        try:
            if hasattr(model_module, self.task):
                create_model_func = getattr(model_module, self.task)
                print("create_model_func success!!!")
                return create_model_func
            else:
                print(f"'create_model' function not found in {module_name}.")
                return None
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None

    def get_model(self):
        self.model = self.model_fuc(self.config)
        return self.model