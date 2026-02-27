# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# train.py
# =============================================================================
#
# 训练入口脚本（PyCharm 直接 Run / 命令行都支持）
#
# 默认逻辑（推荐你日常使用）：
#   - 直接 Run：读取 config_local.py 的配置
#
# 命令行覆盖逻辑（可选）：
#   - 你可以用 --dataset/--model/--iters 等临时覆盖 config_local.py
#
# 训练流程：
#   1) set_seed
#   2) 构建 train/val dataset + dataloader（支持变尺寸 pad_collate）
#   3) build_model（registry 统一管理）
#   4) train_loop（step-based；定期验证；保存 best/last）
#
# =============================================================================

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# 让 from src... 在 PyCharm 和 命令行都工作
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch
from torch.utils.data import DataLoader

import config_local as cfg
from src.datasets.paired_folder_dataset import PairedEdgeFolderDataset
from src.datasets.collate import pad_collate
from src.models.registry import build_model
from src.engine.trainer import train_loop
from src.utils.seed import set_seed


@dataclass
class TrainArgs:
    # 基本选择
    root: str
    dataset: str
    model: str
    device: str
    seed: int

    # 训练超参
    iters: int
    batch: int
    num_workers: int
    lr: float
    weight_decay: float
    log_every: int
    val_every: int
    deep_supervision: bool
    ds_weight: float
    pad_stride: int

    # 输出目录（会自动拼接：out_root/dataset/model/seedX）
    out_dir: str


def _build_default_out_dir() -> str:
    """
    根据 config_local.py 自动生成输出目录结构，避免你每次手写路径。
    """
    return str(
        Path(cfg.paths.out_root)
        / cfg.run.dataset
        / cfg.run.model
        / f"seed{cfg.run.seed}"
    )


def get_default_args_from_config() -> TrainArgs:
    """
    PyCharm 直接 Run 时使用：完全从 config_local.py 读取默认配置。
    """
    return TrainArgs(
        root=cfg.paths.root,
        dataset=cfg.run.dataset,
        model=cfg.run.model,
        device=cfg.run.device,
        seed=cfg.run.seed,
        iters=cfg.train.iters,
        batch=cfg.train.batch,
        num_workers=cfg.train.num_workers,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        log_every=cfg.train.log_every,
        val_every=cfg.train.val_every,
        deep_supervision=cfg.train.deep_supervision,
        ds_weight=cfg.train.ds_weight,
        pad_stride=cfg.train.pad_stride,
        out_dir=_build_default_out_dir(),
    )


def get_parser() -> argparse.ArgumentParser:
    """
    命令行覆盖参数：不想在 config_local.py 改的时候可以临时用。
    """
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None, choices=["hed"])  # 训练目前只针对深度模型
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--deep_supervision", action="store_true")
    p.add_argument("--no_deep_supervision", action="store_true")
    p.add_argument("--ds_weight", type=float, default=None)
    p.add_argument("--pad_stride", type=int, default=None)

    p.add_argument("--out_dir", type=str, default=None)
    return p


def merge_cli_over_config(args: TrainArgs, ns: argparse.Namespace) -> TrainArgs:
    """
    将命令行参数（若提供）覆盖到默认 args 上。
    这样你既能“统一管理”，也能“临时改一次不污染 config_local.py”。
    """
    for k, v in vars(ns).items():
        if v is None:
            continue
        if k in ("deep_supervision", "no_deep_supervision"):
            continue
        if hasattr(args, k):
            setattr(args, k, v)

    # deep supervision 的开关需要单独处理
    if getattr(ns, "no_deep_supervision", False):
        args.deep_supervision = False
    elif getattr(ns, "deep_supervision", False):
        args.deep_supervision = True

    # out_dir 如果没给，按（out_root/dataset/model/seed）自动拼接
    if args.out_dir is None or str(args.out_dir).strip() == "":
        args.out_dir = _build_default_out_dir()

    return args


def main(args: TrainArgs) -> None:
    # -------------------------------------------------------------------------
    # 0) 复现性：固定随机种子
    # -------------------------------------------------------------------------
    set_seed(args.seed)

    # -------------------------------------------------------------------------
    # 1) Dataset / DataLoader
    #    - train：shuffle=True
    #    - val  ：batch=1 最稳（避免大量 padding 造成浪费）
    # -------------------------------------------------------------------------
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
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # 2) Build model
    #    - hed：torch.nn.Module
    # -------------------------------------------------------------------------
    model = build_model(name=args.model)

    # -------------------------------------------------------------------------
    # 3) 训练主循环（step-based）
    # -------------------------------------------------------------------------
    train_loop(
        model=model,
        dl_train=dl_train,
        dl_val=dl_val,
        device=args.device,
        out_dir=args.out_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        iters=args.iters,
        log_every=args.log_every,
        val_every=args.val_every,
        pad_stride=args.pad_stride,
        deep_supervision=args.deep_supervision,
        ds_weight=args.ds_weight,
    )


if __name__ == "__main__":
    args = get_default_args_from_config()

    # 命令行覆盖（可选）
    if len(sys.argv) > 1:
        parser = get_parser()
        ns = parser.parse_args()
        args = merge_cli_over_config(args, ns)

    main(args)