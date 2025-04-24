import numpy as np
import matplotlib.pyplot as plt
import os
import re

# for i in range(610):
#     file_name = f'/data/yaozhi/segmentation/data/UPENN_GBM/UPENN-GBM/NIfTI-files/automated_segm/UPENN-GBM-{i+1:05d}_11_automated_approx_segm.nii.gz'
#     image_obj = nib.load(file_name)
#     image_data = image_obj.get_fdata().astype(np.int32).reshape(-1)
#     print(np.bincount(image_data))

# file_name = f'/data/yaozhi/segmentation/data/UPENN_GBM/UPENN-GBM/NIfTI-files/automated_segm/UPENN-GBM-00001_11_automated_approx_segm.nii.gz'
# file_name = '/data/yaozhi/segmentation/data/UPENN_GBM/UPENN-GBM-NIfTI/UPENN-GBM/Processed-files/flair_imgs/UPENN-GBM-00002_11_FLAIR.npy'
# image_data = np.load(file_name)
# plt.imsave('image.png', image_data[:, :, 40], cmap='gray')
# file_name = '/data/yaozhi/segmentation/data/UPENN_GBM/UPENN-GBM-NIfTI/UPENN-GBM/Processed-files/masks/UPENN-GBM-00002_11_mask.npy'
# image_data = np.load(file_name).reshape(-1)
# print(np.bincount(image_data))
# plt.imsave('mask.png', image_data[:, :, 40], cmap='gray')

folder_path = '/data/yaozhi/segmentation/data/UPENN_GBM/UPENN-GBM-NIfTI/UPENN-GBM/Processed-files/masks/'
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
sorted_masks = sorted(
    npy_files,
    key=lambda x: int(re.search(r"UPENN-GBM-(\d+)_", x).group(1))
)
print(len(sorted_masks))
for i in range(10):
    file = np.load(folder_path+sorted_masks[i])#.astype(np.int32)
    # print(np.bincount(file.reshape(-1)))
    plt.imsave(f'mask_{i}.png', file[:, :, 40], cmap='gray')

folder_path = '/data/yaozhi/segmentation/data/UPENN_GBM/UPENN-GBM-NIfTI/UPENN-GBM/Processed-files/flair_imgs/'
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
sorted_files = sorted(
    npy_files,
    key=lambda x: int(re.search(r"UPENN-GBM-(\d+)_", x).group(1))
)
print(len(sorted_files))
for i in range(10):
    file = np.load(folder_path+sorted_files[i])#.astype(np.int32)
    # print(np.bincount(file.reshape(-1)))
    plt.imsave(f'image_{i}.png', file[:, :, 40], cmap='gray')