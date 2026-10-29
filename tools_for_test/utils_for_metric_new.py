# -*- coding: utf-8 -*-
"""
Time    : 2024/12/26 15:48
Author  : yuchuang
desc    :
"""

import os
import cv2
import numpy as np
import sys
from skimage import measure
import os
from os.path import join, isfile
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score




def cal_target_AP(precision_list, recall_list):
    avg_precision_target = 0

    for i in range(len(precision_list) - 1):
        avg_precision_target += precision_list[i] * (recall_list[i] - recall_list[i + 1])

    return avg_precision_target

def cal_target_AP_AUC(precision_list, recall_list):
    target_AP_AUC = auc(recall_list, precision_list)
    return target_AP_AUC



def get_centroid(contour):
    """计算轮廓的质心坐标"""
    M = cv2.moments(contour)
    if M['m00'] != 0:
        centroid_x = M['m10'] / M['m00']
        centroid_y = M['m01'] / M['m00']
    else:
        centroid_x = contour[0][0][0]
        centroid_y = contour[0][0][1]
    return np.array([centroid_y, centroid_x])  # 统一格式，方便计算

def cal_true_pred_right_num_new(input_pred_mask, input_true_mask, use_th):
    """
    计算真实目标、预测目标和正确检测目标数量
    质心点相差 < 3 视为正确检测
    """
    # 处理预测掩码
    y_pred_binary = np.where(input_pred_mask > use_th, 255, 0).astype(np.uint8)

    # 获取真实目标的轮廓
    contours_true, _ = cv2.findContours(input_true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    true_target_num = len(contours_true)

    # 获取预测目标的轮廓
    contours_pred, _ = cv2.findContours(y_pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_target_num = len(contours_pred)

    pred_right_centroid_list = []  # 存储匹配成功的预测目标
    matched_pred_indices = set()  # 记录已匹配的预测目标索引

    # 遍历所有真实目标
    for contour_true in contours_true:
        centroid_true = get_centroid(contour_true)

        for i, contour_pred in enumerate(contours_pred):
            if i in matched_pred_indices:
                continue  # 跳过已匹配的预测目标

            centroid_pred = get_centroid(contour_pred)
            distance = np.linalg.norm(centroid_true - centroid_pred)

            if distance < 3:
                pred_right_centroid_list.append(centroid_pred)  # 记录匹配成功的预测质心
                matched_pred_indices.add(i)  # 标记该预测已匹配
                break  # 该真实目标已匹配一个预测目标，退出内层循环

    right_target_num = len(pred_right_centroid_list)

    return true_target_num, pred_target_num, right_target_num







def cal_piexl_results(model_out_path, true_path):
    mask_list = os.listdir(true_path)
    y_pred_out = np.array([])
    y_true_out = np.array([])

    for i in range(len(mask_list)):
        #print("正在处理：", mask_list[i])
        input_mask_name = mask_list[i]
        input_mask_path = os.path.join(model_out_path, input_mask_name)
        input_true_path = os.path.join(true_path, input_mask_name)

        y_pred = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)/255
        input_true = cv2.imread(input_true_path, cv2.IMREAD_GRAYSCALE)/255
        y_true = np.where(input_true > 0.5, 1, 0)
        #print(np.unique(input_mask))

        y_pred = y_pred.ravel()
        y_true = y_true.ravel()
        # print(y_pred.shape)
        # print(y_pred_out.shape)
        # y_pred_out = np.concatenate(y_pred_out, y_pred)
        # y_true_out = np.concatenate(y_true_out, y_true)
        y_pred_out = np.append(y_pred_out, y_pred)
        y_true_out = np.append(y_true_out, y_true)

    #print(y_pred_out.shape)

    # print(np.unique(y_pred_out))
    # print(len(np.unique(y_pred_out)))
    unique_values, counts = np.unique(y_pred_out*255, return_counts=True)
    value_counts = dict(zip(unique_values, counts))
    print("Value counts:", value_counts)
    print(y_true_out.shape)

    # 获取ROC曲线所需的fpr, tpr和阈值
    fpr, tpr, roc_thresholds = roc_curve(y_true_out.ravel(), y_pred_out.ravel())
    roc_auc = auc(fpr, tpr)

    # 获取PR曲线所需的precision, recall和thresholds
    precision, recall, pr_thresholds = precision_recall_curve(y_true_out.ravel(), y_pred_out.ravel())
    average_precision = average_precision_score(y_true_out.ravel(), y_pred_out.ravel())
    average_precision2 = auc(recall, precision)

    # 打印 ROC 和 PR 曲线的 AUC 值
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision (AP): {average_precision:.4f}")
    print(f"Average Precision AUC (AP-AUC): {average_precision2:.4f}")

    return roc_auc, average_precision, average_precision2




def cal_piexl_results_fast(model_out_path, true_path):
    mask_list = os.listdir(true_path)
    # y_pred_out = np.array([])
    # y_true_out = np.array([])
    y_pred_list = []
    y_true_list = []

    for i in range(len(mask_list)):
        #print("正在处理：", mask_list[i])
        input_mask_name = mask_list[i]
        input_mask_path = os.path.join(model_out_path, input_mask_name)
        input_true_path = os.path.join(true_path, input_mask_name)

        y_pred = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)/255
        input_true = cv2.imread(input_true_path, cv2.IMREAD_GRAYSCALE)/255
        y_true = np.where(input_true > 0.5, 1, 0)
        #print(np.unique(input_mask))

        # y_pred = y_pred.ravel()
        # y_true = y_true.ravel()
        # # print(y_pred.shape)
        # # print(y_pred_out.shape)
        # # y_pred_out = np.concatenate(y_pred_out, y_pred)
        # # y_true_out = np.concatenate(y_true_out, y_true)
        # y_pred_out = np.append(y_pred_out, y_pred)
        # y_true_out = np.append(y_true_out, y_true)

        y_pred_list.append(y_pred.ravel())
        y_true_list.append(y_true.ravel())

    #print(y_pred_out.shape)
    y_pred_out = np.concatenate(y_pred_list)
    y_true_out = np.concatenate(y_true_list)

    # print(np.unique(y_pred_out))
    # print(len(np.unique(y_pred_out)))
    unique_values, counts = np.unique(y_pred_out*255, return_counts=True)
    value_counts = dict(zip(unique_values, counts))
    print("Value counts:", value_counts)
    print(y_true_out.shape)

    # 获取ROC曲线所需的fpr, tpr和阈值
    fpr, tpr, roc_thresholds = roc_curve(y_true_out.ravel(), y_pred_out.ravel())
    roc_auc = auc(fpr, tpr)

    # 获取PR曲线所需的precision, recall和thresholds
    precision, recall, pr_thresholds = precision_recall_curve(y_true_out.ravel(), y_pred_out.ravel())
    average_precision = average_precision_score(y_true_out.ravel(), y_pred_out.ravel())
    average_precision2 = auc(recall, precision)

    # 打印 ROC 和 PR 曲线的 AUC 值
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision (AP): {average_precision:.4f}")
    print(f"Average Precision AUC (AP-AUC): {average_precision2:.4f}")

    return roc_auc, average_precision, average_precision2




# def cal_target_results_rethink(model_out_path, true_path, th_list):
#     mask_list = os.listdir(true_path)
#     precision_list = [0]
#     recall_list = [1]
#     for use_th in th_list:
#         print("正在处理阈值：", use_th)
#         true_target_num_total = 0
#         pred_target_num_total = 0
#         right_target_num_total = 0
#         for m in range(len(mask_list)):
#             img_name = mask_list[m]
#             input_true_mask_path = os.path.join(true_path, img_name)
#             input_pred_mask_path = os.path.join(model_out_path, img_name)
#             input_true_mask = cv2.imread(input_true_mask_path, cv2.IMREAD_GRAYSCALE)
#             input_pred_mask = cv2.imread(input_pred_mask_path, cv2.IMREAD_GRAYSCALE)
#             true_target_num, pred_target_num, right_target_num = cal_true_pred_right_num_new(input_pred_mask, input_true_mask, use_th)
#
#             true_target_num_total = true_target_num_total + true_target_num
#             pred_target_num_total = pred_target_num_total + pred_target_num
#             right_target_num_total = right_target_num_total + right_target_num
#
#         precision_out = right_target_num_total/pred_target_num_total
#         recall_out = right_target_num_total/true_target_num_total
#         precision_list.append(precision_out)
#         recall_list.append(recall_out)
#
#     precision_list.append(1)
#     recall_list.append(0)
#
#     print("----------------")
#     print("precision_list:", precision_list)
#     print("recall_list", recall_list)
#     target_AP = cal_target_AP(precision_list, recall_list)
#     # target_AP_AUC = cal_target_AP_AUC(precision_list, recall_list)
#     print("target_AP:", target_AP)
#     # print("target_AP_AUC:", target_AP_AUC)
#     print("----------------")
#     return target_AP


##解决前面可能存在的pred_target_num_total == 0问题
def cal_target_results_rethink(model_out_path, true_path, th_list):
    mask_list = os.listdir(true_path)
    precision_list = [0]
    recall_list = [1]
    for use_th in th_list:
        print("正在处理阈值：", use_th)
        true_target_num_total = 0
        pred_target_num_total = 0
        right_target_num_total = 0
        for m in range(len(mask_list)):
            img_name = mask_list[m]
            input_true_mask_path = os.path.join(true_path, img_name)
            input_pred_mask_path = os.path.join(model_out_path, img_name)
            input_true_mask = cv2.imread(input_true_mask_path, cv2.IMREAD_GRAYSCALE)
            input_pred_mask = cv2.imread(input_pred_mask_path, cv2.IMREAD_GRAYSCALE)
            true_target_num, pred_target_num, right_target_num = cal_true_pred_right_num_new(input_pred_mask, input_true_mask, use_th)

            true_target_num_total = true_target_num_total + true_target_num
            pred_target_num_total = pred_target_num_total + pred_target_num
            right_target_num_total = right_target_num_total + right_target_num

        if pred_target_num_total == 0:
            precision_out = 0
        else:
            precision_out = right_target_num_total/pred_target_num_total
        recall_out = right_target_num_total/true_target_num_total
        precision_list.append(precision_out)
        recall_list.append(recall_out)

    precision_list.append(1)
    recall_list.append(0)

    print("----------------")
    print("precision_list:", precision_list)
    print("recall_list", recall_list)
    target_AP = cal_target_AP(precision_list, recall_list)
    # target_AP_AUC = cal_target_AP_AUC(precision_list, recall_list)
    print("target_AP:", target_AP)
    # print("target_AP_AUC:", target_AP_AUC)
    print("----------------")
    return target_AP






















