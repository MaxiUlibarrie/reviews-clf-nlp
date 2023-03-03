import argparse

from train import TrainModel

from config import Config

config = Config()

def get_opt(config):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version-model",
        type=str,
        required=True,
        default='0',
        help="Version of model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        default=config.get.model.epochs,
        help="number of epochs"
    )

    parser.add_argument(
        "--n-classes",
        type=int,
        required=False,
        default=config.get.model.n_classes,
        help="number of classes"
    )

    parser.add_argument(
        "--max-length-tokens",
        type=int,
        required=False,
        default=config.get.model.max_length_tokens,
        help="Max length of the tokens to be used"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=config.get.model.batch_size,
        help="BATCH_SIZE for training"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        required=False,
        default=config.get.model.random_seed,
        help="random seed for to start training"
    )

    opt, _ = parser.parse_known_args()

    return opt

if __name__ == "__main__":
    opt = get_opt(config)

    train_model = TrainModel(random_seed=opt.random_seed, 
                             max_length_tokens=opt.max_length_tokens, 
                             batch_size=opt.batch_size, 
                             n_classes=opt.n_classes, 
                             epochs=opt.epochs,
                             version_model=opt.version_model)
    
    train_model.train()
