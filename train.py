# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
import sys
from dataclasses import dataclass
from pathlib import Path
import argparse

# -----------------------------------------------------------------------------
# 同 infer.py：保证 from src... 在 PyCharm 和 命令行都工作
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch
from torch.utils.data import DataLoader

from src.datasets.paired_folder_dataset import PairedEdgeFolderDataset
from src.datasets.collate import pad_collate
from src.models.registry import build_model
from src.engine.trainer import train_loop
from src.utils.seed import set_seed


@dataclass
class TrainArgs:
    root: str
    dataset: str
    device: str = "cuda"
    model: str = "hed"

    # 训练超参（先给一个能跑通的默认）
    batch: int = 1
    num_workers: int = 4
    lr: float = 1e-4
    iters: int = 2000
    log_every: int = 50
    val_every: int = 200
    seed: int = 0

    # 输出
    out_dir: str = "outputs/tmp/train"


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model", type=str, default="hed", choices=["hed"])  # 先只做 HED
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    return p


def get_default_args() -> TrainArgs:
    """
    PyCharm直接Run时用这套默认参数。
    """
    root = r"D:\study\project\Data\A1\augmented"
    dataset = "PASD150_0"
    out_dir = r"D:\study\project\ImageDetection\PASD_benchmark\outputs\PASD150_0\hed\seed0"
    return TrainArgs(root=root, dataset=dataset, out_dir=out_dir, model="hed", device="cuda", batch=1)


def main(args: TrainArgs):
    set_seed(args.seed)

    # 1) Dataset / DataLoader
    ds_train = PairedEdgeFolderDataset(args.root, args.dataset, "train")
    ds_val = PairedEdgeFolderDataset(args.root, args.dataset, "val")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=1,              # val/test 用 1 最稳（避免大量padding）
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=args.device.startswith("cuda"),
    )

    # 2) Build model（HED 目前是骨架，等你接入真实实现后才能训练）
    model = build_model(name=args.model)

    # 3) 进入通用训练循环
    train_loop(
        model=model,
        dl_train=dl_train,
        dl_val=dl_val,
        device=args.device,
        out_dir=args.out_dir,
        lr=args.lr,
        iters=args.iters,
        log_every=args.log_every,
        val_every=args.val_every,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = get_parser()
        ns = parser.parse_args()
        args = TrainArgs(**vars(ns))
    else:
        args = get_default_args()

    main(args)

