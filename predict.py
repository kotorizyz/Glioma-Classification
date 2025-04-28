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
from dataset.upenn import upenn

torch.set_printoptions(sci_mode=False)

gpus = [6]

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

def evaluate_segmentation(output, ground_truth, num_classes):
    """
    output: Tensor of shape [BATCH_SIZE, NUM_CLASSES, HEIGHT, WIDTH] (raw logits or probs)
    ground_truth: Tensor of shape [BATCH_SIZE, HEIGHT, WIDTH] (class indices 0~NUM_CLASSES-1)
    num_classes: int, total number of classes
    """

    pred = torch.argmax(output, dim=1)  # [BATCH_SIZE, HEIGHT, WIDTH]

    correct = (pred == ground_truth).float()
    pixel_accuracy = correct.sum() / correct.numel()

    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (ground_truth == cls)
        intersection = (pred_cls & gt_cls).sum().float()
        union = (pred_cls | gt_cls).sum().float()
        if union < 1:
            iou = torch.tensor(1.0, device=output.device)
        else:
            iou = intersection / union
        iou_list.append(iou)

    mean_iou = torch.mean(torch.stack(iou_list)) if iou_list else torch.tensor(0.0)

    return pixel_accuracy.item(), mean_iou.item()


def main():
    args, config = parse_args_and_config()
    model = Model(config).to(device=config.device)
    model = torch.nn.DataParallel(model, device_ids=gpus)
    # load which model
    model.load_state_dict(torch.load('pth/upenn_50.pth', map_location=config.device))
    model.eval()
    
    dataset = upenn(train=False).all_data
    dataloader = Data.DataLoader(dataset=dataset, batch_size=config.training.batch_size, shuffle=True)
    
    pixel_accuracy = 0
    mean_iou = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            images = data[:,:1].to(device=config.device) #[BATCH, CHANNEL, HEIGHT, WIDTH]
            predict_masks = model(images).to(torch.float32) #[BATCH, NUM_CLASSES, HEIGHT, WIDTH]
            ground_truth = data[:, 1].to(device=config.device).to(torch.long)
            pixel_accuracy_i, mean_iou_i = evaluate_segmentation(predict_masks, ground_truth, config.model.out_ch)
            pixel_accuracy += pixel_accuracy_i
            mean_iou += mean_iou_i
            # print(mean_iou_i)
        pixel_accuracy /= len(dataloader)
        mean_iou /= len(dataloader)
        print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    sys.exit(main())