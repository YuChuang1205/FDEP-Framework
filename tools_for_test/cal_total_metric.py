# -*- coding: utf-8 -*-
"""
Time    : 2025/3/10 17:04
Author  : yuchuang
desc    :
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from utils_for_metric_new import *

def make_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)
    else:
        pass

root_path = os.path.abspath('.')
input_data_path = os.path.join(root_path, 'dataset')

dataset_name = 'SIRST3'   ## choose dataset in [SIRST3, IRSTD_1K_point, NUDT_SIRST_1_1_point, SIRST_1_1_point_new]

true_path = os.path.join(input_data_path, dataset_name, 'val', 'mask')
model_out_path = os.path.join(input_data_path, dataset_name, 'val', 'pre_results_no_binary')




th_list = [i * 255 * 0.1 for i in range(10)]
# th_list = [i for i in range(255)]


print("点图处理完毕，开始计算像素级指标--------------")
pixel_roc, pixel_ap, pixel_ap_auc = cal_piexl_results_fast(model_out_path, true_path)
print("像素级指标计算完毕，开始计算目标级指标------------")
target_ap = cal_target_results_rethink(model_out_path, true_path, th_list)
print("目标级指标计算完毕!!!!!!!!!!!!!!!!!!!!!")
print("最终结果输出---------------------------------------")
print("像素级ROC-AUC:", pixel_roc)
# print("像素级AP:", pixel_ap)
print("像素级AP-AUC:", pixel_ap_auc)
print("目标级AP:", target_ap)
print("最终结果(像素级AP-AUC 乘以 目标级AP-AUC):", pixel_ap_auc * target_ap)

print("Done!!!!!")
print("Done!!!!!")
print("Done!!!!!")
