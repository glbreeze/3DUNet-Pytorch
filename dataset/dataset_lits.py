from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
import torch
import nrrd
import h5py
import nibabel as nib
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, Normalize, CropToFixed


class CT_Dataset(dataset):
    def __init__(self, args, mode='train'):

        self.args = args
        self.mode = mode

        if mode == 'train':
            self.filename_list = self.load_file_name_list('./dataset/data_list/train_list.txt')
            self.transforms = Compose([
                # Normalize(mean=stat['mean'], std=stat['std']),
                CropToFixed(size=self.args.crop_size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])
        else:
            self.filename_list = self.load_file_name_list('./dataset/data_list/val_list.txt')
            self.transforms = None
            # self.transforms = Compose([Center_Crop(base=16, max_size=args.val_crop_max_size)])

    def __getitem__(self, index):
        file_name = self.filename_list[index]
        if not file_name.endswith('hdf5'):
            file_name += '.hdf5'

        p_path = os.path.join(self.args.dataset_path, file_name)
        with h5py.File(p_path, 'r') as f:
            img = f['ct']['image'][:]
            seg = f['ct']['seg'][:]

        ct_array = self.normalise_zero_one(img)
        seg_array = seg.astype(np.int32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    @staticmethod
    def normalise_zero_one(image):
        """Image normalisation. Normalises image to fit [0, 1] range."""

        image = image.astype(np.float32)

        minimum = np.min(image)
        maximum = np.max(image)

        if maximum > minimum:
            ret = (image - minimum) / (maximum - minimum)
        else:
            ret = image * 0.
        return ret

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines)
        return file_name_list


def calculate_stats(images, global_normalization=True):
    """
    Calculates min, max, mean, std given a list of nd-arrays
    """
    if global_normalization:
        # flatten first since the images might not be the same size
        flat = np.concatenate(
            [img.ravel() for img in images]
        )
        pmin, pmax, mean, std = np.percentile(flat, 1), np.percentile(flat, 99.6), np.mean(flat), np.std(flat)
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {
        'pmin': pmin,
        'pmax': pmax,
        'mean': mean,
        'std': std
    }

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = CT_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())