import numpy as np
import hdf5storage
from torch.utils.data import Dataset
import torch
import os
import re

class upenn(Dataset):
    def __init__(self, train=True):
        folder_path_masks = '/data/yaozhi/segmentation/data/UPENN_GBM/UPENN-GBM-NIfTI/UPENN-GBM/Processed-files/masks/'
        npy_files = [f for f in os.listdir(folder_path_masks) if f.endswith('.npy')]
        sorted_masks = sorted(
            npy_files,
            key=lambda x: int(re.search(r"UPENN-GBM-(\d+)_", x).group(1))
        )
        folder_path_files = '/data/yaozhi/segmentation/data/UPENN_GBM/UPENN-GBM-NIfTI/UPENN-GBM/Processed-files/flair_imgs/'
        npy_files = [f for f in os.listdir(folder_path_files) if f.endswith('.npy')]
        sorted_files = sorted(
            npy_files,
            key=lambda x: int(re.search(r"UPENN-GBM-(\d+)_", x).group(1))
        )

        
        if train:#130*155
            num_patients = 130
        else:
            num_patients = 17
        self.all_data = torch.zeros(num_patients*155, 2, 240, 240)

        for i in range(num_patients):
            j = i
            if train == False:
                i += num_patients
            file = np.load(folder_path_files+sorted_files[i]).transpose(2, 0, 1)
            self.all_data[j*155:(j+1)*155, 0, :, :] = torch.from_numpy(file)
            mask = np.load(folder_path_masks+sorted_masks[i]).transpose(2, 0, 1)
            self.all_data[j*155:(j+1)*155, 1, :, :] = torch.from_numpy(mask)

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)