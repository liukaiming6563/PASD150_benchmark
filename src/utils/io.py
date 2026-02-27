# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def save_prob_png(prob: torch.Tensor, out_path: str | Path):
    """
    保存边缘概率图：
      输入 prob: torch float in [0,1]
      输出 png: uint8 in [0,255]
    支持形状：
      (H,W) / (1,H,W) / (B,1,H,W)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if prob.ndim == 4:
        prob = prob[0, 0]
    elif prob.ndim == 3:
        prob = prob[0]

    arr = prob.detach().cpu().clamp(0, 1).numpy()
    img = (arr * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)