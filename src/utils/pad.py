# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
import torch
import torch.nn.functional as F
from typing import Tuple

def pad_to_stride(x: torch.Tensor, stride: int = 32, mode: str = "replicate") -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """
    推理/训练时，有些模型希望 H/W 是 stride(常见32) 的倍数。
    这里对 (B,C,H,W) 做 padding，返回 pad 信息用于 crop 回去。

    pad 格式：(left, right, top, bottom)
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride

    left, top = 0, 0
    right, bottom = pad_w, pad_h

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)

    x_pad = F.pad(x, (left, right, top, bottom), mode=mode)
    return x_pad, (left, right, top, bottom)

def crop_back(x: torch.Tensor, pad: Tuple[int,int,int,int]) -> torch.Tensor:
    left, right, top, bottom = pad
    if (left, right, top, bottom) == (0,0,0,0):
        return x
    h, w = x.shape[-2], x.shape[-1]
    return x[..., top:h-bottom, left:w-right]