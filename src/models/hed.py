# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
import torch
import torch.nn as nn

class HEDWrapper(nn.Module):
    """
    HED 包装器（骨架占位）：

    统一约定：forward 输出 logits (B,1,H,W)，不要 sigmoid
    - 训练：BCEWithLogitsLoss
    - 推理：infer_engine 里统一 sigmoid

    你确定使用哪个 HED repo 后，我会把这里替换为“可训练可推理”的真实实现。
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError(
            "HEDWrapper is a stub now. "
            "Tell me which PyTorch HED repo you use (or ask me to implement a minimal HED), "
            "then I'll fill this in."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError




