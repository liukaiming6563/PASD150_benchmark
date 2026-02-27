# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
import torch
import torch.nn as nn

def bce_logits_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """
    logits: (B,1,H,W) 未sigmoid
    target: (B,1,H,W) 0/1
    valid : (B,1,H,W) 0/1，用于忽略pad区域（可选）

    返回：标量 loss
    """
    loss_map = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if valid is not None:
        # 只在有效区域计算平均
        denom = valid.sum().clamp_min(1.0)
        return (loss_map * valid).sum() / denom
    return loss_map.mean()
