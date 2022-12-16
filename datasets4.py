import random
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from typing import Callable, List, Union, Iterable, Tuple, NewType

import numpy
import cv2


# from cartex import expect_valid_float_image
# from cartex import iterativeLPF
# from cartex import LTV, channelwiseLTV
# from cartex import CartoonTextureDecomposition


class TrainDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale
        self.lr_name=[os.path.join('data\DIV2K\\train\LBP-lr',i) for i in os.listdir('data\DIV2K\\train\LBP-lr')]
        self.hr_name=[os.path.join('data\DIV2K\\train\LBP-HR',i) for i in os.listdir('data\DIV2K\\train\LBP-HR')]
    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)][::]
            hr = f['hr'][str(idx)][::]
            lr_texture=cv2.imread(self.lr_name[idx])
            hr_texture=cv2.imread(self.hr_name[idx])



            #            B,C,W,H=hr.shape
            #            hr_texture = np.zeros((B, C, H, W))
            #            for i in range(B):
            #                out_l =hr[i, :, :, :]
            #                gray = cv2.cvtColor(out_l, cv2.COLOR_RGB2GRAY)
            #                out_l = rotation_invariant_LBP(gray)
            #                hr_texture[i, :, :, :] = out_l
            #
            #            a,_,c,d=lr.shape
            #            lr_texture = torch.rand(a, 1, c, d)
            #            for i in range(a):
            #                x_out = lr[i, :, :, :]
            #                gray = cv2.cvtColor(x_out, cv2.COLOR_RGB2GRAY)
            #                x_out = rotation_invariant_LBP(gray)
            #                lr_texture[i, :, :, :] = x_out

            lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
            lr, hr = self.random_horizontal_flip(lr, hr)
            lr, hr = self.random_vertical_flip(lr, hr)
            lr, hr = self.random_rotate_90(lr, hr)
            lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255.0
            hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255.0
            lr_texture, hr_texture = self.random_crop(lr_texture, hr_texture, self.patch_size, self.scale)
            lr_texture, hr_texture = self.random_horizontal_flip(lr_texture, hr_texture)
            lr_texture, hr_texture = self.random_vertical_flip(lr_texture, hr_texture)
            lr_texture, hr_texture = self.random_rotate_90(lr_texture, hr_texture)
            lr_texture = lr_texture.astype(np.float32).transpose([2, 0, 1])
            hr_texture = hr_texture.astype(np.float32).transpose([2, 0, 1])

            return lr_texture, hr_texture,lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.lr_name = [os.path.join('data\DIV2K\evalue\set5_LBP_lr', i) for i in os.listdir('data\DIV2K\evalue\set5_LBP_lr')]
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][str(idx)][::]
            hr = f['hr'][str(idx)][::]
            lr_texture = cv2.imread(self.lr_name[idx])
            # lr_texture = np.transpose(lr_texture, (0, 3, 1, 2))
            # kernel_size = (5, 5)
            #            sigma = 1.5
            #            lr_cartoon = cv2.GaussianBlur(lr, kernel_size, sigma)
            #            lr_texture = lr - lr_cartoon
            #            a,_,c,d=lr.shape
            #            lr_texture = torch.rand(a, 1, c, d)
            #            for i in range(a):
            #                x_out = lr[i, :, :, :]
            #                gray = cv2.cvtColor(x_out, cv2.COLOR_RGB2GRAY)
            #                x_out = rotation_invariant_LBP(gray)
            #                lr_texture[i, :, :, :] = x_out
            # obj = CartoonTextureDecomposition(sigma=2, ksize=7)
            # lr_cartoon, lr_texture = obj.decompose(lr)

            # lr_cartoon = lr_cartoon.astype(np.float32).transpose([2, 0, 1]) / 255.0
            lr_texture = lr_texture.astype(np.float32).transpose([2, 0, 1]) / 255.0
            lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255.0
            hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255.0
            return lr_texture,lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
