# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# infer.py
# =============================================================================
#
# 推理入口脚本（PyCharm 直接 Run / 命令行都支持）
#
# 默认（推荐）：
#   - 直接 Run：读取 config_local.py
#   - 输出目录自动拼接到：out_root/dataset/model/seedX/preds
#
# 输出风格（论文展示）：
#   - invert=True  => 白底黑边
#   - threshold=0.3~0.5 => 更“干净”的二值边缘
#
# =============================================================================

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


@dataclass
class InferArgs:
    root: str
    dataset: str
    split: str
    out_dir: str

    model: str
    device: str

    batch: int
    num_workers: int
    pad_stride: int

    # 论文展示输出风格
    invert: bool
    threshold: float | None

    # Canny 参数
    canny_low: int
    canny_high: int

    # 可选：深度模型 ckpt（不填则不加载）
    ckpt_path: str | None = None


def _build_default_out_dir() -> str:
    """
    自动拼接推理输出目录：out_root/dataset/model/seedX/preds
    """
    return str(
        Path(cfg.paths.out_root)
        / cfg.run.dataset
        / cfg.run.model
        / f"seed{cfg.run.seed}"
        / "preds"
    )


def _build_default_ckpt_path() -> str:
    """
    自动拼接 best checkpoint 路径：
      out_root/dataset/model/seedX/checkpoints/best.pt
    """
    return str(
        Path(cfg.paths.out_root)
        / cfg.run.dataset
        / cfg.run.model
        / f"seed{cfg.run.seed}"
        / "checkpoints"
        / "best.pt"
    )


def get_default_args_from_config() -> InferArgs:
    """
    PyCharm 直接 Run 时使用：完全从 config_local.py 读取默认配置。
    """
    # 默认推理时，如果是深度模型（hed），我们倾向加载 best.pt（若存在）
    ckpt = _build_default_ckpt_path() if cfg.run.model in ("hed",) else None

    return InferArgs(
        root=cfg.paths.root,
        dataset=cfg.run.dataset,
        split=cfg.run.split,
        out_dir=_build_default_out_dir(),
        model=cfg.run.model,
        device=cfg.run.device,
        batch=cfg.infer.batch,
        num_workers=cfg.infer.num_workers,
        pad_stride=cfg.infer.pad_stride,
        invert=cfg.infer.invert,
        threshold=cfg.infer.threshold,
        canny_low=cfg.canny.low,
        canny_high=cfg.canny.high,
        ckpt_path=ckpt,
    )


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--model", type=str, default=None, choices=["canny", "hed"])
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--pad_stride", type=int, default=None)

    p.add_argument("--invert", action="store_true")
    p.add_argument("--no_invert", action="store_true")
    p.add_argument("--threshold", type=float, default=None)

    p.add_argument("--canny_low", type=int, default=None)
    p.add_argument("--canny_high", type=int, default=None)

    p.add_argument("--ckpt_path", type=str, default=None)
    return p


def merge_cli_over_config(args: InferArgs, ns: argparse.Namespace) -> InferArgs:
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

    # 如果 out_dir 未指定，保持默认拼接
    if args.out_dir is None or str(args.out_dir).strip() == "":
        args.out_dir = _build_default_out_dir()

    return args


def main(args: InferArgs) -> None:
    # 1) Dataset / DataLoader
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

    # 2) Build model
    model = build_model(
        name=args.model,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )

    # 3) 如果是 torch 模型且提供 ckpt，则加载
    if isinstance(model, torch.nn.Module) and args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            # 兼容你 trainer 保存格式：{"model": state_dict}
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state, strict=True)
            print(f"[infer] loaded ckpt: {ckpt_path}")
        else:
            print(f"[infer][warn] ckpt not found, will run with current weights: {ckpt_path}")

    # 4) 推理并保存（输出白底黑边由 invert 控制）
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