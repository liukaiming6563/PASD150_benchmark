# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
from src.models.canny import CannyEdge
from src.models.hed import HEDWrapper

def build_model(name: str, **kwargs):
    """
    统一模型构建入口：
    - name="canny" 返回 CannyEdge（非torch模型）
    - name="hed"   返回 HEDWrapper（torch模型）

    以后加 RCF/BDCN/PiDiNet/TEED 就在这里扩展即可
    """
    name = name.lower().strip()
    if name == "canny":
        low = kwargs.get("canny_low", 50)
        high = kwargs.get("canny_high", 150)
        return CannyEdge(low=low, high=high)

    if name == "hed":
        return HEDWrapper()

    raise ValueError(f"Unknown model name: {name}")
