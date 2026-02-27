# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
import sys
from dataclasses import dataclass
from pathlib import Path
import argparse

# -----------------------------------------------------------------------------
# 让 "from src...." 在 PyCharm 直接 Run/Debug 和 命令行两边都能稳定工作
# 原理：把本文件所在目录（PASD_benchmark）加入 sys.path
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch
from torch.utils.data import DataLoader

from src.datasets.paired_folder_dataset import PairedEdgeFolderDataset
from src.datasets.collate import pad_collate
from src.models.registry import build_model
from src.engine.infer_engine import infer_and_save


@dataclass
class InferArgs:
    # 数据根目录：D:\study\project\Data\A1\augmented
    root: str
    # 数据集名：BIPED / BSDS500 / Multicue_0 / ... / PASD150_2
    dataset: str
    # split: train/val/test
    split: str = "test"
    # 输出目录：保存预测边缘图
    out_dir: str = "outputs/tmp/preds"
    # batch size（变尺寸情况下建议 val/test 用 1）
    batch: int = 1
    num_workers: int = 4

    # 模型名：canny/hed/...
    model: str = "canny"
    # 设备：cpu/cuda
    device: str = "cpu"
    # pad_to_stride（推理时为避免shape不对齐）
    pad_stride: int = 32

    # Canny 参数（如果 model=canny 会用到）
    canny_low: int = 50
    canny_high: int = 150


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--model", type=str, default="canny",
                   choices=["canny", "hed"])  # 先支持这俩，后面再加 rcf/bdcn/pidinet/teed
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--pad_stride", type=int, default=32)

    p.add_argument("--canny_low", type=int, default=50)
    p.add_argument("--canny_high", type=int, default=150)
    return p


def get_default_args() -> InferArgs:
    """
    PyCharm 直接点 Run 时会走这里（不需要敲命令行）。
    你想切换数据集/输出目录，直接改这里即可。
    """
    root = r"D:\study\project\Data\A1\augmented"
    dataset = "BIPED"
    split = "test"
    out_dir = r"D:\study\project\ImageDetection\PASD_benchmark\outputs\BIPED\canny\preds_test"
    return InferArgs(root=root, dataset=dataset, split=split, out_dir=out_dir, model="canny", device="cpu")


def main(args: InferArgs):
    # 1) 构建 dataset / dataloader（val/test 通常 batch=1 最省心）
    ds = PairedEdgeFolderDataset(args.root, args.dataset, args.split)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,  # 支持变尺寸 batch
        pin_memory=(args.device.startswith("cuda")),
    )

    # 2) 构建模型（canny 是非 torch 模型；hed 是 torch 模型）
    model = build_model(
        name=args.model,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )

    # 3) 推理并保存预测图
    infer_and_save(
        model=model,
        loader=dl,
        out_dir=args.out_dir,
        device=args.device,
        pad_stride=args.pad_stride,
    )

    print(f"[DONE] Saved predictions to: {args.out_dir}")


if __name__ == "__main__":
    # 兼容：PyCharm直接Run（无参数） vs 命令行（有参数）
    if len(sys.argv) > 1:
        parser = get_parser()
        ns = parser.parse_args()
        args = InferArgs(**vars(ns))
    else:
        args = get_default_args()

    main(args)

