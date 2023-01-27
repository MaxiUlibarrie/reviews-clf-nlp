import json

CONFIG_PATH = "/usr/src/config.json"

class Config():

    def __init__(self):

        with open(CONFIG_PATH) as config_file:
            config_json = config_file.read()

        config = json.loads(config_json)

        self.batch_size = int(config["model"]["batch-size"])
        self.max_length_tokens = int(config["model"]["max-length-tokens"])
        self.random_seed = int(config["model"]["random-seed"])
        self.epochs = int(config["model"]["epochs"])
        self.n_classes = int(config["model"]["n-classes"])

        print("### Config loaded ###")
        