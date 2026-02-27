# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.utils.pad import pad_to_stride, crop_back
from src.utils.io import save_prob_png


@torch.no_grad()
def infer_and_save(model, loader: DataLoader, out_dir: str | Path, device: str = "cuda", pad_stride: int = 32):
    """
    通用推理入口：兼容两类模型
    1) torch.nn.Module：forward 输出 logits (B,1,H,W)
       推理时 sigmoid -> prob
    2) 非 torch 模型（Canny）：提供 model.infer(img_chw)->(1,H,W) prob

    loader 输出由 pad_collate 提供：
      imgs:  (B,3,H,W) float [0,1]
      edges: (B,1,H,W) float {0,1}  # 推理时不用
      valid: (B,1,H,W)             # 推理时不用
      metas: list[dict]            # 主要用 filename 保存预测
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_torch = isinstance(model, torch.nn.Module)
    if is_torch:
        model.eval().to(device)

    for imgs, edges, valid, metas in loader:
        if is_torch:
            imgs = imgs.to(device)

            # pad-to-stride，防止某些模型对齐出错
            x, pad = pad_to_stride(imgs, stride=pad_stride, mode="replicate")

            logits = model(x)                 # (B,1,H',W')
            logits = crop_back(logits, pad)   # (B,1,H,W)
            prob = torch.sigmoid(logits)      # (B,1,H,W)
        else:
            # Canny 逐张推理（支持 batch）
            prob_list = []
            for i in range(imgs.shape[0]):
                prob_list.append(model.infer(imgs[i]))  # (1,H,W)
            prob = torch.stack(prob_list, dim=0)         # (B,1,H,W)

        # 按原文件名保存预测图
        for i, meta in enumerate(metas):
            save_prob_png(prob[i], out_dir / meta["filename"])