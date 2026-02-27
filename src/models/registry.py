# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/models/registry.py
# =============================================================================
#
# 通过字符串名称构建模型，避免 train.py / infer.py 到处写 if/else。
#
# 目前支持：
#   - canny：传统算法（非 nn.Module），提供 infer(img_chw)->(1,H,W) prob
#   - hed  ：深度模型（nn.Module），forward 输出 logits 或 (fuse, sides)
#
# =============================================================================

from __future__ import annotations

from src.models.canny import CannyEdgeDetector
from src.models.hed import HEDWrapper


def build_model(name: str, **kwargs):
    """
    Args:
        name: 模型名称字符串（不区分大小写）
        kwargs: 不同模型的额外参数
            - canny: canny_low, canny_high
    """
    name = name.lower().strip()

    if name == "canny":
        low = int(kwargs.get("canny_low", 50))
        high = int(kwargs.get("canny_high", 150))
        return CannyEdgeDetector(low_threshold=low, high_threshold=high)

    if name == "hed":
        # HED：使用 torchvision 的 VGG16 backbone
        # pretrained=True：自动下载/缓存 ImageNet 权重（你已验证可用）
        return HEDWrapper(pretrained=True, return_sides_in_train=True)

    raise ValueError(f"Unknown model name: {name}")