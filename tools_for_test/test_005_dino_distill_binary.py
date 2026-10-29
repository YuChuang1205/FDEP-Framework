#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time :
@desc:
"""
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
# from model.MSDA.MSDA_no_sigmoid import MSDANet_No_Sigmoid
from skimage import measure
import torch.nn.functional as F
from torch.autograd import Variable
import math
from components.cal_mean_std import Calculate_mean_std
import importlib
import sys
from utilts import *

def read_txt(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    image_out_list = [line.strip() + '.png' for line in lines]
    return image_out_list


def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

##动态调取模型 (升级版)  在原始版本上加了dino的测试版本
def access_model_dino(choose_model, model_name_num):
    choose_model_dir_name = choose_model + '_no_sigmoid' + '_' + model_name_num
    model_function = choose_model + '_No_Sigmoid'
    module_name = f"model.{choose_model}.{choose_model_dir_name}"  # 完整模块路径
    module = importlib.import_module(module_name)  # 导入模块
    model_func = getattr(module, model_function)  # 从模块中获取指定的函数
    return model_func

##############################################
choose_model = 'MSDA_dino'   ##choose model in [ACM, ALC, MLCL, ALCL, DNA, GGL, UIU, MSDA]
model_name_num = '337'
model_func = access_model_dino(choose_model,model_name_num)
choose_dataset = 'IRSTD_1K_point'  ## choose dataset in [SIRST3, IRSTD_1K_point, NUDT_SIRST_1_1_point, SIRST_1_1_point_new]
test_dir_name = 'demo017_013__2025-03-28_10-35-16'  ##输入对应的文件测试模型所在的文件夹名
test_model_name = 'best_mIoU_checkpoint_' + test_dir_name + ".pth.tar"
image_deal_scheme = 'crop_scheme'  ##choose one in ['resize_scheme', 'crop_scheme'] 如果选择'resize_scheme'，需保证resize的尺寸比PATCH_SIZE_test小
################################################



# Hyperparameters etc.
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
IMAGE_RESIZE_SIZE = 448
TEST_PATCH_SIZE = 448
PATCH_SIZE_min = 112   
patch_size_test = 1024  
TEST_PATCH_BATCH_SIZE = 4
root_path = os.path.abspath('.')
dataset_path = os.path.join(root_path,'dataset', choose_dataset)
test_dataset_path = os.path.join(dataset_path,'val')
input_path = os.path.join(test_dataset_path, 'img')
output_path = os.path.join(test_dataset_path, 'pre_results')
make_dir(output_path)
# TEST_NUM = len(os.listdir(input_path))
# txt_path = os.path.join(root_path, 'img_idx', 'test.txt')
img_list = os.listdir(input_path)

test_model_path = os.path.join(root_path,'work_dirs',test_dir_name,test_model_name)

def test_pred(img, net, batch_size, patch_size):
    b, c, h, w = img.shape
    # print(img.shape)
    patch_size = patch_size
    stride = patch_size

    if h > patch_size or w > patch_size:
        # Unfold the image into patches
        img_unfold = F.unfold(img, kernel_size=patch_size, stride=stride)
        img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        # print(img_unfold.shape)
        patch_num = img_unfold.size(1)

        preds_list = []
        for i in range(0, patch_num, batch_size):
            end = min(i + batch_size, patch_num)
            batch_patches = img_unfold[:, i:end, :, :, :].reshape(-1, c, patch_size, patch_size)
            batch_patches = Variable(batch_patches.float())
            if choose_model == 'DNA_dino' or choose_model == 'UIU_dino':
                batch_preds = net.forward(batch_patches)
                preds_list.append(batch_preds)
            else:
                batch_preds,_,_ = net.forward(batch_patches)
                preds_list.append(batch_preds)
        # Concatenate all the patch predictions
        preds_unfold = torch.cat(preds_list, dim=0).permute(1, 2, 3, 0)
        preds_unfold = preds_unfold.reshape(b, -1, patch_num)
        preds = F.fold(preds_unfold, kernel_size=patch_size, stride=stride, output_size=(h, w))
    else:
        if choose_model == 'DNA_dino' or choose_model == 'UIU_dino':
            preds = net.forward(img)
        else:
            preds,_,_ = net.forward(img)
    return preds


class SirstDataset(Dataset):
    def __init__(self, image_dir, patch_size, patch_size_min, transform=None, mode='None', image_deal_scheme = 'resize_scheme'):
        self.image_dir = image_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))
        self.mode = mode
        self.patch_size = patch_size
        self.patch_size_min = patch_size_min
        self.image_deal_scheme = image_deal_scheme

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w, _ = image.shape

        if (self.mode == 'test'):
            if image_deal_scheme == 'crop_scheme':
                times = self.patch_size_min
                h, w, c = image.shape
                h1 = h
                w1 = w
                if min(h, w) < self.patch_size:
                    image = np.pad(image, ((0, max(h, self.patch_size) - h), (0, max(w, self.patch_size) - w), (0, 0)),
                                   mode='constant')
                    h, w, _ = image.shape

                pad_height = math.ceil(h / times) * times - h
                pad_width = math.ceil(w / times) * times - w
                image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            elif image_deal_scheme == 'resize_scheme':
                pass
            else:
                print("image_deal_scheme的赋值有错误，请在['resize_scheme', 'crop_scheme']中选择！！！")
                sys.exit(0)
            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations["image"]
            return image, self.images[index], h1, w1
        else:
            print("输入的模式错误！！！")


def main():
    origin_img_dir = dataset_path + "/origin/img"
    cal_mean, cal_std = Calculate_mean_std(origin_img_dir)

    if image_deal_scheme == 'crop_scheme':
        test_transforms = A.Compose(
            [
                #A.Resize(height=IMAGE_RESIZE_SIZE, width=IMAGE_RESIZE_SIZE),
                A.Normalize(
                    mean=cal_mean,
                    std=cal_std,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
    elif image_deal_scheme == 'resize_scheme':
        test_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_RESIZE_SIZE, width=IMAGE_RESIZE_SIZE),
                A.Normalize(
                    mean=cal_mean,
                    std=cal_std,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
    else:
        print("image_deal_scheme的赋值有错误，请在['resize_scheme', 'crop_scheme']中选择！！！")
        sys.exit(0)

    test_ds = SirstDataset(
        image_dir=input_path,
        patch_size=TEST_PATCH_SIZE,
        patch_size_min=PATCH_SIZE_min,
        transform=test_transforms,
        mode='test',
        image_deal_scheme=image_deal_scheme
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    model = model_func().to(DEVICE)

    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load(test_model_path, map_location=DEVICE)[
                               'state_dict'].items()}, strict=False)
    model.eval()

    temp_num = 0

    for idx, (img, name, h, w) in enumerate(test_loader):
        print(idx)
        img = img.to(device=DEVICE)
        with torch.no_grad():
            image_1 = img

            output_1 = test_pred(image_1, model, batch_size=TEST_PATCH_BATCH_SIZE, patch_size=patch_size_test)
            if image_deal_scheme == 'crop_scheme':
                output_1 = output_1[:, :, :h, :w]
                output_1 = torch.sigmoid(output_1)
            elif image_deal_scheme == 'resize_scheme':
                output_1 = F.interpolate(output_1, size=(h, w), mode='bilinear',
                                       align_corners=False)
                output_1 = torch.sigmoid(output_1)
            else:
                print("image_deal_scheme的赋值有错误，请在['resize_scheme', 'crop_scheme']中选择！！！")
                sys.exit(0)

            output_1 = output_1.cpu().data.numpy()

        for i in range(output_1.shape[0]):
            print(name[i])
            temp_num = temp_num + 1
            pred = output_1[i]
            pred = pred[0]
            pred_target = np.where(pred > 0.5, 255, 0)
            pred_target = np.array(pred_target, dtype='uint8')

            cv2.imwrite(os.path.join(output_path, name[i]), pred_target)


if __name__ == "__main__":
    main()