# -*- coding: utf-8 -*-
"""
Time    : 2026/01
Author  : yuchuang
desc    : total metric (fast evaluation)
"""

import os
from utils_for_metric_new_fast import (
    cal_piexl_results_fast,
    cal_target_results_rethink,
)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

root_path = os.path.abspath('.')
input_data_path = os.path.join(root_path, 'dataset')

dataset_name = 'SIRST3'
true_path = os.path.join(input_data_path, dataset_name, 'val', 'mask')
model_out_path = os.path.join(input_data_path, dataset_name, 'val', 'pre_results_no_binary')

# threshold list (unchanged logic)
th_list = [i * 255 * 0.1 for i in range(10)]

print("点图处理完毕，开始计算像素级指标--------------")
pixel_roc, _, pixel_ap_auc = cal_piexl_results_fast(model_out_path, true_path)

print("像素级指标计算完毕，开始计算目标级指标------------")
target_ap = cal_target_results_rethink(model_out_path, true_path, th_list)

print("目标级指标计算完毕!!!!!!!!!!!!!!!!!!!!!")
print("最终结果输出---------------------------------------")
print("像素级ROC-AUC:", pixel_roc)
print("像素级AP-AUC:", pixel_ap_auc)
print("目标级AP:", target_ap)
print("最终结果(HSE-like):", pixel_ap_auc * target_ap)

print("Done!!!!!")
