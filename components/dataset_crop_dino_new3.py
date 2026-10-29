import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import time
from components.edges import onehot_to_binary_edges, mask_to_onehot
import cv2
import random
import math



def random_crop(img, mask, patch_size,patch_size_min):
    # 确保 img 和 mask 的形状都正确，并且包含三个维度
    h, w, c = img.shape
    mh, mw = mask.shape
    times = patch_size_min

    # 确保 img 和 mask 的高度和宽度相同
    assert (h, w) == (mh, mw), "Image and mask must have the same height and width"

    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w, _ = img.shape

    pad_height = math.ceil(h / times) * times - h
    pad_width = math.ceil(w / times) * times - w
    # 填充图像和掩码
    img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')

    h, w, _ = img.shape

    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size

    img_patch = img[h_start:h_end, w_start:w_end, :]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


class SirstDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size, patch_size_min, transform=None, mode='None'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))
        self.mode = mode
        self.patch_size = patch_size
        self.patch_size_min = patch_size_min

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        # print(image.shape)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # img.convert(‘L’)为灰度图像
        mask = (mask > 127.5).astype(float)

        if (self.mode == 'train'):
            image_patch, mask_patch = random_crop(image, mask, self.patch_size,self.patch_size_min)
            if self.transform is not None:
                augmentations = self.transform(image=image_patch, mask=mask_patch)
                image = augmentations["image"]
                mask = augmentations["mask"]
            mask_2 = mask.numpy()
            mask_2 = mask_2.astype(np.int64)
            oneHot_label = mask_to_onehot(mask_2, 2)
            edge = onehot_to_binary_edges(oneHot_label, 1, 2)
            # 消去图像边缘
            edge[1, :] = 0
            edge[-1:, :] = 0
            edge[:, :1] = 0
            edge[:, -1:] = 0
            edge = np.expand_dims(edge, axis=0).astype(np.int64)
            return image, mask, edge

        elif (self.mode == 'val'):
            mask_original = mask
            times = self.patch_size_min
            h, w, c = image.shape
            # 填充高度和宽度，使其能被32整除
            h1 = h
            w1 = w

            if min(h, w) < self.patch_size:
                image = np.pad(image, ((0, max(h, self.patch_size) - h), (0, max(w, self.patch_size) - w), (0, 0)), mode='constant')
                mask = np.pad(mask, ((0, max(h, self.patch_size) - h), (0, max(w, self.patch_size) - w)), mode='constant')
                h, w, _ = image.shape

            pad_height = math.ceil(h / times) * times - h
            pad_width = math.ceil(w / times) * times - w
            # 填充图像和掩码
            image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            return image, mask, mask_original, h1, w1
