# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/engine/trainer.py
# =============================================================================
#
# 通用训练循环（step-based），便于不同数据集/模型公平比较。
#
# 支持两类 forward 输出：
#   1) logits: Tensor (B,1,H,W)
#   2) (fuse_logits, [side1..side5])：用于 HED 深监督
#
# 注意：
#   - 输入来自 pad_collate：imgs/edges/valid/metas
#   - 训练/验证都会 pad_to_stride，避免尺寸对齐问题
#
# =============================================================================

from pathlib import Path
import torch

from src.engine.losses import bce_logits_loss
from src.utils.pad import pad_to_stride, crop_back


def train_loop(
    model,
    dl_train,
    dl_val,
    device: str,
    out_dir: str,
    lr: float,
    weight_decay: float,
    iters: int,
    log_every: int = 50,
    val_every: int = 200,
    pad_stride: int = 32,
    deep_supervision: bool = True,
    ds_weight: float = 1.0,
):
    """
    Args:
        model: torch.nn.Module
        dl_train/dl_val: DataLoader，输出 (imgs, edges, valid, metas)
        device: "cpu"/"cuda"
        out_dir: 实验输出根目录（会创建 checkpoints）
        lr, weight_decay: optimizer 超参数
        iters: 总训练步数
        log_every/val_every: 日志与验证频率
        pad_stride: 输入 pad 到 stride 倍数
        deep_supervision: 是否对 side outputs 也计算 loss
        ds_weight: 深监督 loss 权重
    """
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 1e9
    step = 0
    train_iter = iter(dl_train)

    while step < iters:
        try:
            imgs, edges, valid, metas = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            imgs, edges, valid, metas = next(train_iter)

        imgs = imgs.to(device)
        edges = edges.to(device)
        valid = valid.to(device)

        # pad-to-stride：避免模型下采样/上采样造成的 shape mismatch
        x, pad = pad_to_stride(imgs, stride=pad_stride, mode="replicate")
        edges_p, _ = pad_to_stride(edges, stride=pad_stride, mode="constant", value=0.0)
        valid_p, _ = pad_to_stride(valid, stride=pad_stride, mode="constant", value=0.0)

        opt.zero_grad(set_to_none=True)

        out = model(x)

        # 兼容 HED 深监督输出
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            fuse_logits, side_logits_list = out[0], out[1]
        else:
            fuse_logits, side_logits_list = out, None

        # crop 回原 padded 前的尺寸（与 edges_p/valid_p 对齐）
        fuse_logits = crop_back(fuse_logits, pad)

        # 主输出 loss（fuse）
        loss = bce_logits_loss(fuse_logits, crop_back(edges_p, pad), crop_back(valid_p, pad), balance=True)

        # 深监督：对每个 side output 也算 loss
        if deep_supervision and side_logits_list is not None:
            side_loss_sum = 0.0
            for s in side_logits_list:
                s = crop_back(s, pad)
                side_loss_sum = side_loss_sum + bce_logits_loss(
                    s, crop_back(edges_p, pad), crop_back(valid_p, pad), balance=True
                )
            loss = loss + ds_weight * (side_loss_sum / float(len(side_logits_list)))

        loss.backward()
        opt.step()

        step += 1

        if step % log_every == 0:
            print(f"[train] step={step}/{iters} loss={loss.item():.6f}")

        if step % val_every == 0:
            val_loss = validate(
                model=model,
                dl_val=dl_val,
                device=device,
                pad_stride=pad_stride,
                deep_supervision=deep_supervision,
                ds_weight=ds_weight,
            )
            print(f"[val] step={step} val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = out_dir / "checkpoints" / "best.pt"
                torch.save({"model": model.state_dict()}, ckpt_path)
                print(f"[ckpt] saved best to {ckpt_path}")

    # 保存最后一次
    ckpt_last = out_dir / "checkpoints" / "last.pt"
    torch.save({"model": model.state_dict()}, ckpt_last)
    print(f"[ckpt] saved last to {ckpt_last}")


@torch.no_grad()
def validate(
    model,
    dl_val,
    device: str,
    pad_stride: int = 32,
    deep_supervision: bool = True,
    ds_weight: float = 1.0,
) -> float:
    """
    验证：返回平均 val_loss（按 batch 平均）。
    """
    model.eval()
    total = 0.0
    n = 0

    for imgs, edges, valid, metas in dl_val:
        imgs = imgs.to(device)
        edges = edges.to(device)
        valid = valid.to(device)

        x, pad = pad_to_stride(imgs, stride=pad_stride, mode="replicate")
        edges_p, _ = pad_to_stride(edges, stride=pad_stride, mode="constant", value=0.0)
        valid_p, _ = pad_to_stride(valid, stride=pad_stride, mode="constant", value=0.0)

        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            fuse_logits, side_logits_list = out[0], out[1]
        else:
            fuse_logits, side_logits_list = out, None

        fuse_logits = crop_back(fuse_logits, pad)

        loss = bce_logits_loss(fuse_logits, crop_back(edges_p, pad), crop_back(valid_p, pad), balance=True)

        if deep_supervision and side_logits_list is not None:
            side_loss_sum = 0.0
            for s in side_logits_list:
                s = crop_back(s, pad)
                side_loss_sum = side_loss_sum + bce_logits_loss(
                    s, crop_back(edges_p, pad), crop_back(valid_p, pad), balance=True
                )
            loss = loss + ds_weight * (side_loss_sum / float(len(side_logits_list)))

        total += float(loss.item())
        n += 1

    model.train()
    return total / max(1, n)