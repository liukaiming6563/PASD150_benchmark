# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
from pathlib import Path
import torch
from src.engine.losses import bce_logits_loss

def train_loop(
    model,
    dl_train,
    dl_val,
    device: str,
    out_dir: str,
    lr: float,
    iters: int,
    log_every: int = 50,
    val_every: int = 200,
):
    """
    通用训练循环（给 HED/其他深度模型用）
    - 以 fixed iters 为训练预算（你之前也倾向这种更公平）
    - val_every 步做一次验证

    注意：目前 HEDWrapper 还是 stub，等你接入真实HED后才能运行。
    """
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

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

        opt.zero_grad(set_to_none=True)
        logits = model(imgs)  # (B,1,H,W)
        loss = bce_logits_loss(logits, edges, valid)
        loss.backward()
        opt.step()

        step += 1

        if step % log_every == 0:
            print(f"[train] step={step}/{iters} loss={loss.item():.6f}")

        if step % val_every == 0:
            val_loss = validate(model, dl_val, device=device)
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
def validate(model, dl_val, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0

    for imgs, edges, valid, metas in dl_val:
        imgs = imgs.to(device)
        edges = edges.to(device)
        valid = valid.to(device)

        logits = model(imgs)
        loss = bce_logits_loss(logits, edges, valid)

        total += float(loss.item())
        n += 1

    model.train()
    return total / max(1, n)
