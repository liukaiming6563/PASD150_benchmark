# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
import torch

def pad_collate(batch):
    """
    batch 内图片尺寸可能不同，无法直接 stack。
    这里做 batch 内 padding，并额外返回 valid_mask（pad区域=0）。

    返回：
      imgs  : (B,3,Hmax,Wmax)
      edges : (B,1,Hmax,Wmax)
      valid : (B,1,Hmax,Wmax)  # 可用于 masked loss
      metas : list[dict]
    """
    imgs, edges, metas = zip(*batch)

    max_h = max(x.shape[1] for x in imgs)
    max_w = max(x.shape[2] for x in imgs)

    b = len(imgs)
    imgs_out = imgs[0].new_zeros((b, 3, max_h, max_w))
    edges_out = edges[0].new_zeros((b, 1, max_h, max_w))
    valid = torch.zeros((b, 1, max_h, max_w), dtype=imgs_out.dtype)

    for i, (im, ed) in enumerate(zip(imgs, edges)):
        h, w = im.shape[1], im.shape[2]
        imgs_out[i, :, :h, :w] = im
        edges_out[i, :, :h, :w] = ed
        valid[i, :, :h, :w] = 1.0

    return imgs_out, edges_out, valid, list(metas)