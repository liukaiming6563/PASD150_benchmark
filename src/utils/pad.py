# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/utils/pad.py
# =============================================================================
#
# pad_to_stride：
#   将输入张量 pad 到 stride 的倍数（常用于 CNN 下采样/上采样对齐）。
#
# crop_back：
#   将输出按 pad 信息裁剪回原始尺寸。
#
# 支持输入形状：
#   - (B,C,H,W)
#
# =============================================================================

from __future__ import annotations
import torch
import torch.nn.functional as torch_func


def pad_to_stride(
    x: torch.Tensor,
    stride: int = 32,
    mode: str = "replicate",
    value: float = 0.0,
):
    """
    Args:
        x: (B,C,H,W)
        stride: 目标对齐倍数
        mode: torch_func.pad 模式（"replicate"/"constant" 等）
        value: constant 模式使用的填充值

    Returns:
        x_pad: pad 后张量
        pad: (pad_left, pad_right, pad_top, pad_bottom)
    """
    b, c, h, w = x.shape
    stride = int(stride)

    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride

    pad_top = 0
    pad_left = 0
    pad_bottom = pad_h
    pad_right = pad_w

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)

    # torch.nn.functional.pad 的顺序是 (left, right, top, bottom)
    x_pad = torch_func.pad(
        x,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode=mode,
        value=value,
    )
    return x_pad, (pad_left, pad_right, pad_top, pad_bottom)


def crop_back(y: torch.Tensor, pad):
    """
    Args:
        y: (B,C,H_pad,W_pad)
        pad: (pad_left, pad_right, pad_top, pad_bottom)

    Returns:
        y_crop: (B,C,H,W)
    """
    pad_left, pad_right, pad_top, pad_bottom = pad
    _, _, h, w = y.shape

    h2 = h - pad_bottom
    w2 = w - pad_right
    return y[:, :, pad_top:h2, pad_left:w2]