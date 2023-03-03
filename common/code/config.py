import json
import os
from types import SimpleNamespace

ENV_VAR_CONFIG_FILE = "CONFIG_FILE"

class Config():
    """
    Retrieves the info from config.json file and it \
    converts it into a class for easy access for all components \
    in the system.

    This is a Singleton class.
    """
    __shared_instance = None

    def __new__(cls):
        if cls.__shared_instance is None:
            cls.__shared_instance = super().__new__(cls)
            cls.get = cls.load_config()
        
        return cls.__shared_instance  

    @classmethod
    def load_config(cls):

        CONFIG_FILE = os.environ.get(ENV_VAR_CONFIG_FILE)

        with open(CONFIG_FILE) as config_file:
            config_json = config_file.read()

        getter = json.loads(config_json, object_hook = lambda d : SimpleNamespace(**d))

        return getter
        