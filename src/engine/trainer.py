# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/engine/trainer.py
# =============================================================================
#
# 通用训练循环（step-based），便于不同数据集/模型公平比较。
#
# ✅ 支持两类 forward 输出：
#   1) logits: Tensor (B,1,H,W)
#   2) (fuse_logits, [side1..side5])：用于 HED 深监督（deep supervision）
#
# ✅ 支持变尺寸 batch：
#   - DataLoader 使用 pad_collate，返回 imgs/edges/valid/metas
#   - 训练/验证进一步 pad_to_stride（默认 stride=32），保证网络下采样/上采样对齐
#
# ✅ Checkpoint 命名（防止 5×6 + cross-dataset 混乱）：
#   - 自描述 best：best__m-{model}__tr-{dataset}__sd-{seed}__it-{step}__vl-{val}.pt
#   - 自描述 last：last__m-{model}__tr-{dataset}__sd-{seed}__it-{iters}.pt
#   - 便捷别名：best.pt / last.pt
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
import torch

from src.engine.losses import bce_logits_loss
from src.utils.pad import pad_to_stride, crop_back


def _safe_name(s: str) -> str:
    """
    把字符串变成适合文件名的形式，避免出现空格/斜杠等导致路径异常。
    例如： "NYUD v2" -> "NYUD_v2"
    """
    s = str(s)
    for ch in [" ", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "_")
    return s


def train_loop(
    model,
    dl_train,
    dl_val,
    device: str,
    out_dir: str,
    lr: float,
    weight_decay: float,
    iters: int,
    exp_model_name: str,
    exp_train_dataset: str,
    exp_seed: int,
    log_every: int = 50,
    val_every: int = 200,
    pad_stride: int = 32,
    deep_supervision: bool = True,
    ds_weight: float = 1.0,
):
    """
    Args:
        model:
            torch.nn.Module。要求 forward 输出 logits 或 (fuse_logits, side_logits_list)
        dl_train/dl_val:
            DataLoader，必须返回 (imgs, edges, valid, metas)
            imgs : (B,3,H,W) float in [0,1]
            edges: (B,1,H,W) float in {0,1}
            valid: (B,1,H,W) float in {0,1}，padding 区域为 0
        device:
            "cpu" / "cuda"
        out_dir:
            实验输出根目录：会创建 out_dir/checkpoints
        lr, weight_decay:
            Adam 的学习率与权重衰减
        iters:
            总训练 step 数
        exp_model_name / exp_train_dataset / exp_seed:
            用于写入 checkpoint 文件名，避免 cross-dataset 混乱
        log_every / val_every:
            打印训练日志、进行验证的频率（按 step）
        pad_stride:
            pad 到 stride 倍数（默认 32）
        deep_supervision:
            True 则对 HED 的 side outputs 也算 loss（更接近原版 HED）
        ds_weight:
            深监督 loss 权重（总 loss = fuse_loss + ds_weight * mean(side_losses)）
    """
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 安全化字符串，确保文件名不会因为特殊字符崩掉
    exp_model_name = _safe_name(exp_model_name)
    exp_train_dataset = _safe_name(exp_train_dataset)
    exp_seed = int(exp_seed)

    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 1e9
    step = 0
    train_iter = iter(dl_train)

    while step < iters:
        # ---------------------------------------------------------------------
        # 1) 取一个 batch（dl_train 是可循环的，这里用手动迭代器写法实现 step-based）
        # ---------------------------------------------------------------------
        try:
            imgs, edges, valid, metas = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            imgs, edges, valid, metas = next(train_iter)

        imgs = imgs.to(device)
        edges = edges.to(device)
        valid = valid.to(device)

        # ---------------------------------------------------------------------
        # 2) pad_to_stride：避免网络多次下采样导致尺寸对不上
        #    - imgs：用 replicate pad（更自然）
        #    - edges/valid：用 constant 0 pad（padding 区域当做“无边缘/无效”）
        # ---------------------------------------------------------------------
        x, pad = pad_to_stride(imgs, stride=pad_stride, mode="replicate")
        edges_p, _ = pad_to_stride(edges, stride=pad_stride, mode="constant", value=0.0)
        valid_p, _ = pad_to_stride(valid, stride=pad_stride, mode="constant", value=0.0)

        opt.zero_grad(set_to_none=True)

        # ---------------------------------------------------------------------
        # 3) 前向传播
        #    - 兼容输出：
        #      a) fuse_logits
        #      b) (fuse_logits, side_logits_list)
        # ---------------------------------------------------------------------
        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            fuse_logits, side_logits_list = out[0], out[1]
        else:
            fuse_logits, side_logits_list = out, None

        # 将预测裁剪回 pad 前的尺寸，保证与 edges/valid 对齐
        fuse_logits = crop_back(fuse_logits, pad)
        edges_c = crop_back(edges_p, pad)
        valid_c = crop_back(valid_p, pad)

        # ---------------------------------------------------------------------
        # 4) Loss（class-balanced BCE，适合边缘检测极不平衡）
        # ---------------------------------------------------------------------
        loss = bce_logits_loss(fuse_logits, edges_c, valid_c, balance=True)

        # 深监督（HED）：side outputs 也算 loss
        if deep_supervision and side_logits_list is not None:
            side_loss_sum = 0.0
            for s in side_logits_list:
                s = crop_back(s, pad)
                side_loss_sum = side_loss_sum + bce_logits_loss(s, edges_c, valid_c, balance=True)
            loss = loss + ds_weight * (side_loss_sum / float(len(side_logits_list)))

        loss.backward()
        opt.step()

        step += 1

        # ---------------------------------------------------------------------
        # 5) 日志
        # ---------------------------------------------------------------------
        if step % log_every == 0:
            print(f"[train] step={step}/{iters} loss={loss.item():.6f}")

        # ---------------------------------------------------------------------
        # 6) 验证 + 保存 best
        # ---------------------------------------------------------------------
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

            # 若本次验证更好：保存“自描述 best” + 更新 best.pt 别名
            if val_loss < best_val:
                best_val = val_loss

                best_name = (
                    f"best__m-{exp_model_name}"
                    f"__tr-{exp_train_dataset}"
                    f"__sd-{exp_seed}"
                    f"__it-{step}"
                    f"__vl-{best_val:.6f}.pt"
                )
                ckpt_path = ckpt_dir / best_name

                # 1) 保存自描述 best（用于长期归档和 cross-dataset）
                torch.save({"model": model.state_dict()}, ckpt_path)

                # 2) 同时更新别名 best.pt（用于“默认加载最优”）
                ckpt_best_alias = ckpt_dir / "best.pt"
                torch.save({"model": model.state_dict()}, ckpt_best_alias)

                print(f"[ckpt] saved best to {ckpt_path}")

    # -------------------------------------------------------------------------
    # 7) 训练结束：保存 last（自描述 last + last.pt 别名）
    # -------------------------------------------------------------------------
    last_name = (
        f"last__m-{exp_model_name}"
        f"__tr-{exp_train_dataset}"
        f"__sd-{exp_seed}"
        f"__it-{iters}.pt"
    )
    ckpt_last = ckpt_dir / last_name
    torch.save({"model": model.state_dict()}, ckpt_last)

    ckpt_last_alias = ckpt_dir / "last.pt"
    torch.save({"model": model.state_dict()}, ckpt_last_alias)

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

    Returns:
        float: mean val loss
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
        edges_c = crop_back(edges_p, pad)
        valid_c = crop_back(valid_p, pad)

        loss = bce_logits_loss(fuse_logits, edges_c, valid_c, balance=True)

        if deep_supervision and side_logits_list is not None:
            side_loss_sum = 0.0
            for s in side_logits_list:
                s = crop_back(s, pad)
                side_loss_sum = side_loss_sum + bce_logits_loss(s, edges_c, valid_c, balance=True)
            loss = loss + ds_weight * (side_loss_sum / float(len(side_logits_list)))

        total += float(loss.item())
        n += 1

    model.train()
    return total / max(1, n)