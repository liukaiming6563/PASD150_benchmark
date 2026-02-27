# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
import numpy as np
import torch
import cv2

class CannyEdge:
    """
    Canny 是传统基线：
    - 不训练
    - 输入 RGB 图（我们用灰度）
    - 输出二值边缘图（0/1），当作概率图保存
    """

    def __init__(self, low: int = 50, high: int = 150):
        self.low = int(low)
        self.high = int(high)

    @torch.no_grad()
    def infer(self, img_chw: torch.Tensor) -> torch.Tensor:
        """
        img_chw: (3,H,W) float [0,1]
        return:  (1,H,W) float [0,1]
        """
        # tensor -> uint8 RGB
        img = (img_chw.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)  # CHW
        img = np.transpose(img, (1, 2, 0))  # HWC

        # rgb -> gray
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # canny
        edge = cv2.Canny(gray, self.low, self.high)  # 0/255

        edge01 = (edge.astype(np.float32) / 255.0)[None, ...]  # (1,H,W)
        return torch.from_numpy(edge01)

