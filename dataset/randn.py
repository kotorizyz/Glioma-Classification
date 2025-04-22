import numpy as np
import hdf5storage
from torch.utils.data import Dataset
import torch

class randn(Dataset):
    def __init__(self):
        self.images = torch.randn(100, 1, 160, 160).to(torch.float32)
        self.masks = torch.randint(0, 3, (100, 1, 160, 160)).to(torch.long)
        self.all_data = torch.cat((self.images, self.masks), dim=1)


    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)