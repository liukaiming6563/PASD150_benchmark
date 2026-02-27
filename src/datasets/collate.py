# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/datasets/collate.py
# =============================================================================
#
# pad_collate：
#   解决 batch 内图片尺寸不同，torch 默认无法 stack 的问题。
#
# 输出：
#   imgs  : (B,3,Hmax,Wmax) float32
#   edges : (B,1,Hmax,Wmax) float32
#   valid : (B,1,Hmax,Wmax) float32  真实区域=1，padding=0，用于 masked loss
#   metas : list[dict]
#
# =============================================================================

import torch


def pad_collate(batch):
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