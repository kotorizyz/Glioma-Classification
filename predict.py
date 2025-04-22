import argparse
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.unet import Model
from dataset.randn import randn

torch.set_printoptions(sci_mode=False)

gpus = [0]

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device(f"cuda:{gpus[0]}") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)

    # torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    model = Model(config).to(device=config.device)
    model = torch.nn.DataParallel(model, device_ids=gpus)
    model.eval()
    
    dataset = randn().all_data
    dataloader = Data.DataLoader(dataset=dataset, batch_size=config.training.batch_size, shuffle=True)
    
    model.train()
    for batch_idx, data in enumerate(dataloader):
        images = data[:,:1].to(device=config.device) #[BATCH, CHANNEL, HEIGHT, WIDTH]
        predict_masks = model(images).to(torch.float32) #[BATCH, NUM_CLASSES, HEIGHT, WIDTH]

if __name__ == "__main__":
    sys.exit(main())