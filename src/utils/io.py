# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/utils/io.py
# =============================================================================
#
# save_prob_png：
#   将概率图 prob（0~1）保存为单通道 PNG（0~255）。
#
# 支持论文展示：
#   - invert=True  => 白底黑边（保存 1-prob）
#   - threshold!=None => 二值化（更干净）
#
# 输入形状支持：
#   (H,W) / (1,H,W) / (B,1,H,W) ——最终都按单张图保存
#
# =============================================================================

from pathlib import Path
import numpy as np
from PIL import Image
import torch


def save_prob_png(
    prob: torch.Tensor,
    out_path: str | Path,
    invert: bool = False,
    threshold: float | None = None,
) -> None:
    """
    Args:
        prob: torch float in [0,1]
        out_path: 输出 png 路径
        invert: True => 白底黑边（保存 1-prob）
        threshold: 若给定（0~1），先二值化再保存（更适合论文展示）
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if prob.ndim == 4:
        prob = prob[0, 0]
    elif prob.ndim == 3:
        prob = prob[0]

    arr = prob.detach().cpu().float().clamp(0, 1).numpy()

    if threshold is not None:
        arr = (arr >= float(threshold)).astype(np.float32)

    if invert:
        arr = 1.0 - arr

    img = (arr * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)