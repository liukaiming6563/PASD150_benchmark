# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 0):
    """
    固定随机种子，保证重复实验可复现（尤其你 Multicue/PASD 有 0/1/2 三份划分）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)