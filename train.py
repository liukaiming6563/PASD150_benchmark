# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# train.py
# =============================================================================
#
# 训练入口（PyCharm Run 优先），所有默认参数统一来自 config_local.py
#
# 运行逻辑：
#   1) set_seed
#   2) 构建 train/val dataset + dataloader（pad_collate 支持变尺寸 batch）
#   3) build_model（registry 统一管理）
#   4) train_loop（step-based；定期 val；保存 best/last）
#
# 注意：
#   - checkpoint 命名包含 model/dataset/seed/step/val_loss，适合 cross-dataset 实验管理
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

from torch.utils.data import DataLoader

import config_local as cfg
from src.datasets.paired_folder_dataset import PairedEdgeFolderDataset
from src.datasets.collate import pad_collate
from src.models.registry import build_model
from src.engine.trainer import train_loop
from src.utils.seed import set_seed


@dataclass
class TrainArgs:
    # 路径与实验组合
    root: str
    dataset: str
    model: str
    device: str
    seed: int
    out_dir: str

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


def _build_default_out_dir(dataset: str, model: str, seed: int) -> str:
    """
    自动输出目录：
        out_root/dataset/model/seedX
    """
    return str(Path(cfg.paths.out_root) / dataset / model / f"seed{seed}")


def get_default_args_from_config() -> TrainArgs:
    """
    PyCharm 直接 Run 时的默认参数：全部来自 config_local.py
    """
    out_dir = _build_default_out_dir(cfg.run.dataset, cfg.run.model, cfg.run.seed)
    return TrainArgs(
        root=cfg.paths.root,
        dataset=cfg.run.dataset,
        model=cfg.run.model,
        device=cfg.run.device,
        seed=cfg.run.seed,
        out_dir=out_dir,
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
    )


def get_parser() -> argparse.ArgumentParser:
    """
    命令行覆盖（可选）：临时改一次不污染 config_local.py
    """
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None, choices=["hed", "rcf"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--pad_stride", type=int, default=None)

    p.add_argument("--deep_supervision", action="store_true")
    p.add_argument("--no_deep_supervision", action="store_true")
    p.add_argument("--ds_weight", type=float, default=None)
    return p


def merge_cli_over_config(args: TrainArgs, ns: argparse.Namespace) -> TrainArgs:
    """
    CLI 覆盖：仅当用户显式传参时覆盖。
    """
    for k, v in vars(ns).items():
        if v is None:
            continue
        if k in ("deep_supervision", "no_deep_supervision"):
            continue
        if hasattr(args, k):
            setattr(args, k, v)

    # deep supervision 开关单独处理
    if getattr(ns, "no_deep_supervision", False):
        args.deep_supervision = False
    elif getattr(ns, "deep_supervision", False):
        args.deep_supervision = True

    # out_dir：若仍为空，自动生成
    if args.out_dir is None or str(args.out_dir).strip() == "":
        args.out_dir = _build_default_out_dir(args.dataset, args.model, args.seed)

    return args


def main(args: TrainArgs) -> None:
    # 0) 固定随机种子
    set_seed(args.seed)

    # 1) 构建 dataset / dataloader
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

    # 2) build model
    model = build_model(name=args.model)

    # 3) train loop（把 exp_* 信息传入，用于 ckpt 自描述命名）
    train_loop(
        model=model,
        dl_train=dl_train,
        dl_val=dl_val,
        device=args.device,
        out_dir=args.out_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        iters=args.iters,
        exp_model_name=args.model,
        exp_train_dataset=args.dataset,
        exp_seed=args.seed,
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