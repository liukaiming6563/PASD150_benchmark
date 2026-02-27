# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/models/canny.py
# =============================================================================
#
# Canny 基线（传统算法，不参与训练）
#
# 输入：
#   img_chw: torch.float32 (3,H,W), [0,1]
#
# 输出：
#   prob: torch.float32 (1,H,W), [0,1]
#   - 这里把 Canny 的二值结果映射为 0/1 概率图，便于统一保存/对比。
#
# =============================================================================

import numpy as np
import cv2
import torch


class CannyEdgeDetector:
    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low = int(low_threshold)
        self.high = int(high_threshold)

    def infer(self, img_chw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_chw: (3,H,W) float32 in [0,1]
        Returns:
            (1,H,W) float32 in [0,1]
        """
        img = img_chw.detach().cpu().float().clamp(0, 1).numpy()  # CHW
        img_hwc = (np.transpose(img, (1, 2, 0)) * 255.0).astype(np.uint8)

        gray = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, self.low, self.high)  # 0/255

        prob = (edge.astype(np.float32) / 255.0)[None, ...]  # (1,H,W)
        return torch.from_numpy(prob)