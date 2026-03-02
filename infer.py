# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# infer.py
# =============================================================================
#
# 推理入口（PyCharm Run / 命令行均支持），默认从 config_local.py 读取参数。
#
# 关键特性：
#   1) 输出目录自动拼接，并包含 split，避免 train/val/test 混在一起：
#        out_root/dataset/model/seedX/preds_{split}/
#   2) 论文展示输出：
#        invert=True  => 白底黑边
#        threshold!=None => 二值化（更干净）
#   3) 自动寻找 checkpoint（用于不传 --ckpt_path 的场景）：
#        checkpoints/best.pt
#        checkpoints/best__*.pt (最新修改的一个)
#        checkpoints/last.pt
#        checkpoints/last__*.pt (最新修改的一个)
#      若都不存在：不会报错，只给 warning，并用当前权重推理。
#
# cross-dataset 推理建议：
#   - 测试数据集由 --dataset 指定
#   - 训练权重来自另一个数据集时，直接用 --ckpt_path 指向对应 best.pt
#
# =============================================================================

from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# 保证 from src... 在 PyCharm 和 命令行都工作
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
from src.engine.infer_engine import infer_and_save


# -----------------------------------------------------------------------------
# 深度模型列表：用于决定“默认是否自动找 ckpt”
# 注意：这里是维护点，后面加模型只需改这一处。
# -----------------------------------------------------------------------------
DEEP_MODELS = ("hed", "rcf", "bdcn", "pidinet", "teed")


@dataclass
class InferArgs:
    # 数据相关
    root: str
    dataset: str
    split: str

    # 输出相关
    out_dir: str

    # 模型/设备/seed
    model: str
    device: str
    seed: int

    # DataLoader
    batch: int
    num_workers: int
    pad_stride: int

    # 输出风格（论文展示）
    invert: bool
    threshold: float | None

    # Canny 参数（仅 model=canny 时使用）
    canny_low: int
    canny_high: int

    # 可选：手动指定 ckpt（cross-dataset 时最常用）
    ckpt_path: str | None = None


# =============================================================================
# 1) 路径拼接工具
# =============================================================================
def _build_default_out_dir(dataset: str, model: str, seed: int, split: str) -> str:
    """
    默认输出目录（包含 split）：
        out_root/dataset/model/seedX/preds_{split}
    """
    return str(Path(cfg.paths.out_root) / dataset / model / f"seed{seed}" / f"preds_{split}")


def _default_ckpt_dir(train_dataset: str, model: str, seed: int) -> Path:
    """
    默认 ckpt 目录：
        out_root/train_dataset/model/seedX/checkpoints

    注意：
        - 这里的 train_dataset 默认取 cfg.run.dataset（即“同数据集推理”）
        - cross-dataset 时建议直接用 --ckpt_path 指定，不走自动发现。
    """
    return Path(cfg.paths.out_root) / train_dataset / model / f"seed{seed}" / "checkpoints"


def _pick_latest(paths: list[Path]) -> Path | None:
    """
    在候选文件中选出“最新修改时间”的那个，用于 best__*.pt / last__*.pt 自动选择。
    """
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def auto_find_ckpt(train_dataset: str, model: str, seed: int) -> str | None:
    """
    自动查找 ckpt 的优先级（从高到低）：
      1) best.pt
      2) best__*.pt 的最新一个
      3) last.pt
      4) last__*.pt 的最新一个

    Returns:
        ckpt_path (str) or None
    """
    ckpt_dir = _default_ckpt_dir(train_dataset, model, seed)
    if not ckpt_dir.exists():
        return None

    best_alias = ckpt_dir / "best.pt"
    if best_alias.exists():
        return str(best_alias)

    best_latest = _pick_latest(list(ckpt_dir.glob("best__*.pt")))
    if best_latest is not None:
        return str(best_latest)

    last_alias = ckpt_dir / "last.pt"
    if last_alias.exists():
        return str(last_alias)

    last_latest = _pick_latest(list(ckpt_dir.glob("last__*.pt")))
    if last_latest is not None:
        return str(last_latest)

    return None


# =============================================================================
# 2) 参数构建：config 默认 + 命令行覆盖
# =============================================================================
def get_default_args_from_config() -> InferArgs:
    """
    PyCharm 直接 Run：默认读取 config_local.py
    """
    out_dir = _build_default_out_dir(cfg.run.dataset, cfg.run.model, cfg.run.seed, cfg.run.split)

    ckpt_path = None
    if cfg.run.model in DEEP_MODELS:
        # 默认：同数据集/同模型/同 seed 的 best/last
        ckpt_path = auto_find_ckpt(cfg.run.dataset, cfg.run.model, cfg.run.seed)

    return InferArgs(
        root=cfg.paths.root,
        dataset=cfg.run.dataset,
        split=cfg.run.split,
        out_dir=out_dir,
        model=cfg.run.model,
        device=cfg.run.device,
        seed=cfg.run.seed,
        batch=cfg.infer.batch,
        num_workers=cfg.infer.num_workers,
        pad_stride=cfg.infer.pad_stride,
        invert=cfg.infer.invert,
        threshold=cfg.infer.threshold,
        canny_low=cfg.canny.low,
        canny_high=cfg.canny.high,
        ckpt_path=ckpt_path,
    )


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # 数据/输出
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, default=None)

    # 模型/设备/seed
    p.add_argument("--model", type=str, default=None, choices=["canny", "hed", "rcf", "bdcn", "pidinet", "teed"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    # loader
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--pad_stride", type=int, default=None)

    # 输出风格
    p.add_argument("--invert", action="store_true")
    p.add_argument("--no_invert", action="store_true")
    p.add_argument("--threshold", type=float, default=None)

    # canny 参数
    p.add_argument("--canny_low", type=int, default=None)
    p.add_argument("--canny_high", type=int, default=None)

    # 手动指定 ckpt（cross-dataset 最常用）
    p.add_argument("--ckpt_path", type=str, default=None)

    return p


def merge_cli_over_config(args: InferArgs, ns: argparse.Namespace) -> InferArgs:
    """
    将命令行参数（若提供）覆盖到默认 args 上。
    """
    # 常规字段覆盖（None 表示不覆盖）
    for k, v in vars(ns).items():
        if v is None:
            continue
        if k in ("invert", "no_invert"):
            continue
        if hasattr(args, k):
            setattr(args, k, v)

    # invert/no_invert 单独处理
    if getattr(ns, "no_invert", False):
        args.invert = False
    elif getattr(ns, "invert", False):
        args.invert = True

    # out_dir 若未指定，则按“当前 args 的 dataset/model/seed/split”自动拼接
    if args.out_dir is None or str(args.out_dir).strip() == "":
        args.out_dir = _build_default_out_dir(args.dataset, args.model, args.seed, args.split)

    # ckpt_path 若未指定：
    #   - 深度模型：自动找同数据集（args.dataset）下的 best/last（同 seed）
    #   - 传统模型：保持 None
    if (args.ckpt_path is None or str(args.ckpt_path).strip() == "") and (args.model in DEEP_MODELS):
        args.ckpt_path = auto_find_ckpt(args.dataset, args.model, args.seed)

    return args


# =============================================================================
# 3) ckpt 加载（安全：找不到不报错）
# =============================================================================
def load_ckpt_if_needed(model, ckpt_path: str | None) -> None:
    """
    - 只对 torch.nn.Module 生效
    - ckpt_path 为空或不存在：不报错，仅提示
    """
    if not isinstance(model, torch.nn.Module):
        return

    if ckpt_path is None or str(ckpt_path).strip() == "":
        print("[infer] no ckpt provided/found, will run with current weights.")
        return

    p = Path(ckpt_path)
    if not p.exists():
        print(f"[infer][warn] ckpt not found: {p} (will run with current weights)")
        return

    ckpt = torch.load(str(p), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"[infer] loaded ckpt: {p}")


# =============================================================================
# 4) 主流程
# =============================================================================
def main(args: InferArgs) -> None:
    # 1) dataset / dataloader
    ds = PairedEdgeFolderDataset(args.root, args.dataset, args.split)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    # 2) build model
    model = build_model(
        name=args.model,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )

    # 3) load ckpt (if needed)
    load_ckpt_if_needed(model, args.ckpt_path)

    # 4) infer and save
    infer_and_save(
        model=model,
        loader=dl,
        out_dir=args.out_dir,
        device=args.device,
        pad_stride=args.pad_stride,
        invert=args.invert,
        threshold=args.threshold,
    )

    print(f"[DONE] Saved predictions to: {args.out_dir}")


if __name__ == "__main__":
    args = get_default_args_from_config()

    # 命令行覆盖（可选）
    if len(sys.argv) > 1:
        parser = get_parser()
        ns = parser.parse_args()
        args = merge_cli_over_config(args, ns)

    main(args)