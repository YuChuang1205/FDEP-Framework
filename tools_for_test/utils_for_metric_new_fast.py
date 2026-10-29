# -*- coding: utf-8 -*-
"""
Time    : 2026/01
Author  : yuchuang (accelerated version)
desc    : logic-preserving fast evaluation
"""

import os
import cv2
import numpy as np
from os.path import join
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ============================================================
# Target-level utilities
# ============================================================

def cal_target_AP(precision_list, recall_list):
    avg_precision_target = 0.0
    for i in range(len(precision_list) - 1):
        avg_precision_target += precision_list[i] * (recall_list[i] - recall_list[i + 1])
    return avg_precision_target


def get_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        cx = contour[0][0][0]
        cy = contour[0][0][1]
    return np.array([cy, cx], dtype=np.float32)


# ============================================================
# Pixel-level (FAST, logic-equivalent)
# ============================================================

def cal_piexl_results_fast(model_out_path, true_path):
    mask_list = os.listdir(true_path)

    y_pred_list = []
    y_true_list = []

    for name in mask_list:
        pred = cv2.imread(join(model_out_path, name), cv2.IMREAD_GRAYSCALE)
        gt   = cv2.imread(join(true_path, name), cv2.IMREAD_GRAYSCALE)

        # keep logic: pred in [0,1], gt in {0,1}
        pred = pred.astype(np.float32) * (1.0 / 255.0)
        gt   = (gt > 127).astype(np.uint8)

        y_pred_list.append(pred.reshape(-1))
        y_true_list.append(gt.reshape(-1))

    y_pred_out = np.concatenate(y_pred_list)
    y_true_out = np.concatenate(y_true_list)

    # ROC
    fpr, tpr, _ = roc_curve(y_true_out, y_pred_out)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_true_out, y_pred_out)
    ap_auc = auc(recall, precision)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Pixel AP-AUC: {ap_auc:.4f}")

    return roc_auc, None, ap_auc


# ============================================================
# Target-level (FAST, logic-equivalent)
# ============================================================

def cal_target_results_rethink(model_out_path, true_path, th_list):
    mask_list = os.listdir(true_path)

    # ---------- preload all masks ----------
    gt_masks = {}
    pred_masks = {}
    gt_centroids = {}

    for name in mask_list:
        gt = cv2.imread(join(true_path, name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(join(model_out_path, name), cv2.IMREAD_GRAYSCALE)

        gt_masks[name] = gt
        pred_masks[name] = pred

        contours, _ = cv2.findContours(
            gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        gt_centroids[name] = [get_centroid(c) for c in contours]

    precision_list = [0.0]
    recall_list = [1.0]

    # ---------- threshold loop ----------
    for use_th in th_list:
        print("正在处理阈值：", use_th)

        true_total = 0
        pred_total = 0
        right_total = 0

        for name in mask_list:
            gt_cs = gt_centroids[name]
            true_total += len(gt_cs)

            pred_bin = np.where(
                pred_masks[name] > use_th, 255, 0
            ).astype(np.uint8)

            contours_pred, _ = cv2.findContours(
                pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            pred_cs = [get_centroid(c) for c in contours_pred]
            pred_total += len(pred_cs)

            # -------- centroid matching (logic unchanged) --------
            matched = set()
            for ct in gt_cs:
                for i, cp in enumerate(pred_cs):
                    if i in matched:
                        continue
                    dx = ct[0] - cp[0]
                    dy = ct[1] - cp[1]
                    if dx * dx + dy * dy < 9.0:  # distance < 3
                        matched.add(i)
                        right_total += 1
                        break

        precision = 0.0 if pred_total == 0 else right_total / pred_total
        recall = 0.0 if true_total == 0 else right_total / true_total

        precision_list.append(precision)
        recall_list.append(recall)

    precision_list.append(1.0)
    recall_list.append(0.0)

    print("----------------")
    print("precision_list:", precision_list)
    print("recall_list:", recall_list)

    target_AP = cal_target_AP(precision_list, recall_list)
    print("target_AP:", target_AP)
    print("----------------")

    return target_AP
