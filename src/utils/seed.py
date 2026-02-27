# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/utils/seed.py
# =============================================================================
#
# 复现性：固定随机种子（Python / NumPy / Torch）
#
# =============================================================================

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Args:
        seed: 任意整数
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了可复现，禁用某些非确定性算法（会略微影响速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False