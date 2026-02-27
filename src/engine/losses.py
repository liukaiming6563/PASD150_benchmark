# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/engine/losses.py
# =============================================================================
#
# 边缘检测的核心问题：正样本（边缘）极少，类别极度不均衡。
# HED 等经典方法通常使用 class-balanced BCE。
#
# bce_logits_loss：
#   - 输入 logits（未 sigmoid）
#   - target 0/1
#   - valid mask 忽略 padding 区域
#
# =============================================================================

import torch
import torch.nn.functional as torch_func


def bce_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    balance: bool = True,
) -> torch.Tensor:
    """
    Args:
        logits: (B,1,H,W) raw logits (no sigmoid)
        target: (B,1,H,W) float {0,1}
        valid : (B,1,H,W) float {0,1}, padding 区域为 0
        balance: True 使用 class-balanced BCE（推荐用于边缘检测）

    Returns:
        scalar loss
    """
    target = target.float()
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = valid.float()

    if not balance:
        loss_map = torch_func.binary_cross_entropy_with_logits(logits, target, reduction="none")
        denom = valid.sum().clamp_min(1.0)
        return (loss_map * valid).sum() / denom

    # class-balanced：beta = |neg| / (|pos|+|neg|)
    with torch.no_grad():
        pos = (target * valid).sum()
        neg = ((1.0 - target) * valid).sum()
        total = (pos + neg).clamp_min(1.0)
        beta = (neg / total).clamp(0.0, 1.0)

    log_p = torch_func.logsigmoid(logits)
    log_not_p = torch_func.logsigmoid(-logits)

    loss_map = -(beta * target * log_p + (1.0 - beta) * (1.0 - target) * log_not_p)
    denom = valid.sum().clamp_min(1.0)
    return (loss_map * valid).sum() / denom