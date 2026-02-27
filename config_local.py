# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# config_local.py
# =============================================================================
#
# 【唯一参数入口】建议你以后跑任何实验只改这一份文件。
#
# 设计目标：
#   1) 训练/推理共用同一套路径、数据集、模型名称、超参数
#   2) PyCharm 直接 Run：train.py / infer.py 自动读取这里的默认值
#   3) 命令行可临时覆盖（可选）：不影响本文件作为“统一配置源”
#
# 目录约定（你现有的数据组织方式）：
#   root/
#     DATASET_NAME/
#       image/{train|val|test}/*.png
#       edge/{train|val|test}/*.png   (同名，GT：黑底白边，0/255)
#
# 输出目录约定（本框架自动拼接）：
#   out_root/
#     DATASET_NAME/
#       MODEL_NAME/
#         seed{seed}/
#           checkpoints/
#           preds/
#
# =============================================================================

from dataclasses import dataclass


# =============================================================================
# 1) 路径相关配置
# =============================================================================
@dataclass
class Paths:
    """
    所有与文件系统路径相关的配置，集中在这里修改即可。
    """

    # 数据根目录（包含所有数据集子文件夹）
    # 例：D:\study\project\Data\A1\augmented
    root: str = r"D:\study\project\Data\A1\augmented"

    # 所有实验输出的根目录（训练日志、ckpt、预测图都写到这里）
    # 例：D:\study\project\PASD150_benchmark\output
    out_root: str = r"D:\study\project\PASD150_benchmark\output"


# =============================================================================
# 2) 本次运行选择（数据集/模型/设备/随机种子）
# =============================================================================
@dataclass
class Run:
    """
    控制“你现在要跑哪个实验组合”。
    """

    # 数据集名（必须对应 root 下的子文件夹名）
    # 例："PASD150_0", "BIPED", "BSDS500"
    dataset: str = "PASD150_0"

    # 训练/推理使用的 split
    # - train.py 会固定用 train + val（这里主要用于 infer.py 默认值）
    split: str = "test"

    # 模型名（必须在 src/models/registry.py 注册）
    # 目前支持： "canny", "hed"
    model: str = "hed"

    # 随机种子：影响可复现性（shuffle、初始化等）
    seed: int = 0

    # 设备： "cpu" / "cuda"
    device: str = "cuda"


# =============================================================================
# 3) 训练超参数（train.py 使用）
# =============================================================================
@dataclass
class Train:
    """
    训练相关的超参数。你后面跑 5×6 基准时，只改这里即可。
    """

    # 总训练步数（step-based 预算，方便不同数据集公平比较）
    iters: int = 2000

    # 训练 batch size（可变尺寸时建议小一点）
    batch: int = 1

    # DataLoader 的 worker 数（Windows 一般 2~4 比较稳）
    num_workers: int = 4

    # Adam 学习率
    lr: float = 1e-4

    # L2 正则（weight decay）
    weight_decay: float = 1e-4

    # 每多少 step 打印一次 train loss
    log_every: int = 50

    # 每多少 step 验证一次（并保存 best）
    val_every: int = 200

    # 是否启用 HED 的深监督（side outputs 也算 loss）
    # 建议：True（更接近原版 HED）
    deep_supervision: bool = True

    # 深监督权重（总 loss = fuse_loss + ds_weight * mean(side_losses)）
    ds_weight: float = 1.0

    # pad 到 stride 倍数（避免下采样/上采样尺寸不对齐）
    pad_stride: int = 32


# =============================================================================
# 4) 推理超参数（infer.py 使用）
# =============================================================================
@dataclass
class Infer:
    """
    推理与保存输出的控制参数。
    """

    # 推理 batch size：变尺寸时设 1 最省心
    batch: int = 1

    num_workers: int = 4

    # 同训练：pad 到 stride 倍数
    pad_stride: int = 32

    # 【论文展示风格】是否反相输出
    # True：白底黑边（你想要的）
    # False：黑底白边
    invert: bool = True

    # 【论文展示风格】是否二值化
    # None：保存灰度概率图（0~255）
    # 0.3~0.5：阈值化后更“黑、更干净”
    threshold: float | None = 0.5


# =============================================================================
# 5) Canny 参数（仅 model=canny 时生效）
# =============================================================================
@dataclass
class Canny:
    # Canny 双阈值
    low: int = 50
    high: int = 150


# =============================================================================
# 实例化：train.py / infer.py 直接 import 这些对象即可
# =============================================================================
paths = Paths()
run = Run()
train = Train()
infer = Infer()
canny = Canny()