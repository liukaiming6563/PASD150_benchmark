# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# infer.py
# =============================================================================
#
# 推理入口（PyCharm Run 优先），默认从 config_local.py 读参数。
#
# 关键特性：
#   ✅ 统一输出目录：out_root/dataset/model/seedX/preds
#   ✅ 论文展示输出：invert=True => 白底黑边；threshold 可二值化
#   ✅ 自动寻找 checkpoint（用于 cross-dataset 推理非常方便）：
#       1) checkpoints/best.pt
#       2) checkpoints/best__*.pt 里“最新的一个”（按文件修改时间）
#       3) checkpoints/last.pt
#       4) checkpoints/last__*.pt 里“最新的一个”
#       都没有则不加载（用当前权重推理）
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
from src.engine.infer_engine import infer_and_save


@dataclass
class InferArgs:
    # 数据相关
    root: str
    dataset: str
    split: str

    # 输出相关
    out_dir: str

    # 模型与设备
    model: str
    device: str

    # loader
    batch: int
    num_workers: int
    pad_stride: int

    # 输出风格（论文展示）
    invert: bool
    threshold: float | None

    # canny 参数
    canny_low: int
    canny_high: int

    # 可选：手动指定 ckpt 路径（如果不指定就自动找 best/last）
    ckpt_path: str | None = None


def _build_default_out_dir(dataset: str, model: str, seed: int) -> str:
    """
    out_root/dataset/model/seedX/preds
    """
    return str(Path(cfg.paths.out_root) / dataset / model / f"seed{seed}" / "preds")


def _default_ckpt_dir(train_dataset: str, model: str, seed: int) -> Path:
    """
    注意：推理时你可能想用“某个训练数据集”的权重去推“另一个测试数据集”。
    这里的 ckpt_dir 默认来自 config_local 的 run.dataset/run.model/run.seed，
    你后续做 cross-dataset 时可以在 CLI 里用 --ckpt_path 直接指定别的训练权重。
    """
    return Path(cfg.paths.out_root) / train_dataset / model / f"seed{seed}" / "checkpoints"


def _pick_latest(glob_list: list[Path]) -> Path | None:
    """
    从候选文件中选出“最新修改”的那个。
    """
    if not glob_list:
        return None
    return max(glob_list, key=lambda p: p.stat().st_mtime)


def auto_find_ckpt(train_dataset: str, model: str, seed: int) -> str | None:
    """
    自动寻找 checkpoint 的策略（按优先级）：
      1) best.pt
      2) best__*.pt 的最新一个
      3) last.pt
      4) last__*.pt 的最新一个
    """
    ckpt_dir = _default_ckpt_dir(train_dataset, model, seed)
    if not ckpt_dir.exists():
        return None

    best_alias = ckpt_dir / "best.pt"
    if best_alias.exists():
        return str(best_alias)

    best_candidates = list(ckpt_dir.glob("best__*.pt"))
    best_latest = _pick_latest(best_candidates)
    if best_latest is not None:
        return str(best_latest)

    last_alias = ckpt_dir / "last.pt"
    if last_alias.exists():
        return str(last_alias)

    last_candidates = list(ckpt_dir.glob("last__*.pt"))
    last_latest = _pick_latest(last_candidates)
    if last_latest is not None:
        return str(last_latest)

    return None


def get_default_args_from_config() -> InferArgs:
    """
    PyCharm 直接 Run：默认读取 config_local.py
    """
    out_dir = _build_default_out_dir(cfg.run.dataset, cfg.run.model, cfg.run.seed)

    # 默认 ckpt：如果是深度模型，则自动找本实验（run.dataset/run.model/run.seed）下的 best/last
    ckpt_path = None
    DEEP_MODELS = ("hed", "rcf", "bdcn", "pidinet", "teed")
    if cfg.run.model in DEEP_MODELS:
        ckpt_path = auto_find_ckpt(cfg.run.dataset, cfg.run.model, cfg.run.seed)

    return InferArgs(
        root=cfg.paths.root,
        dataset=cfg.run.dataset,
        split=cfg.run.split,
        out_dir=out_dir,
        model=cfg.run.model,
        device=cfg.run.device,
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
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--model", type=str, default=None, choices=["canny","hed","rcf", "bdcn", "pidinet"])
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--pad_stride", type=int, default=None)

    p.add_argument("--invert", action="store_true")
    p.add_argument("--no_invert", action="store_true")
    p.add_argument("--threshold", type=float, default=None)

    p.add_argument("--canny_low", type=int, default=None)
    p.add_argument("--canny_high", type=int, default=None)

    # cross-dataset 推理时，最常用的就是显式指定训练权重
    p.add_argument("--ckpt_path", type=str, default=None)
    return p


def merge_cli_over_config(args: InferArgs, ns: argparse.Namespace) -> InferArgs:
    """
    CLI 覆盖：仅当用户显式传参时覆盖。
    """
    for k, v in vars(ns).items():
        if v is None:
            continue
        if k in ("invert", "no_invert"):
            continue
        if hasattr(args, k):
            setattr(args, k, v)

    if getattr(ns, "no_invert", False):
        args.invert = False
    elif getattr(ns, "invert", False):
        args.invert = True

    # out_dir：若仍为空，自动生成
    if args.out_dir is None or str(args.out_dir).strip() == "":
        args.out_dir = _build_default_out_dir(cfg.run.dataset, cfg.run.model, cfg.run.seed)

    return args


def load_ckpt_if_needed(model, ckpt_path: str | None) -> None:
    """
    如果 model 是 torch 模型并且 ckpt_path 存在，则加载权重。
    """
    if not isinstance(model, torch.nn.Module):
        return
    if ckpt_path is None:
        print("[infer] no ckpt provided, will run with current weights.")
        return

    p = Path(ckpt_path)
    if not p.exists():
        print(f"[infer][warn] ckpt not found: {p} (will run with current weights)")
        return

    ckpt = torch.load(str(p), map_location="cpu")
    # 兼容 {"model": state_dict} 或直接 state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"[infer] loaded ckpt: {p}")


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

    # 3) load ckpt (if torch model)
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

    if len(sys.argv) > 1:
        parser = get_parser()
        ns = parser.parse_args()
        args = merge_cli_over_config(args, ns)

    main(args)