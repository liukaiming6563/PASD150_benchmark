# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/models/rcf.py
# =============================================================================
#
# RCF (Rich Convolutional Features) - Simplified Baseline Implementation
# -----------------------------------------------------------------------------
# 说明（非常重要，与你论文表述一致）：
#   本实现是 “RCF-style baseline / RCF variant”，用于统一 benchmark。
#   它保留了 RCF 的核心思想：
#     1) VGG16 多层特征（multi-level features）
#     2) 每层一个 side output + 深监督（deep supervision）
#     3) 所有 side outputs 融合输出（fusion）
#
# 与严格复现的差异：
#   - 严格 RCF 可能会利用更多中间 conv 层并做更复杂的融合，本实现采用
#     “每个 stage 取最后一层特征 + refine block + side” 的简化形式。
#   - 对于你的论文目标（跨数据集 benchmark、公平对比、PASD150 优势证明），
#     这个实现更易维护、更稳定、也更公平。
#
# 输入/输出约定（必须与 engine 对齐）：
#   - 输入 x : (B,3,H,W) float32 in [0,1]
#   - 输出 logits（未 sigmoid）：
#       eval/infer: fuse_logits (B,1,H,W)
#       train: (fuse_logits, [side1..side5]) if return_sides_in_train=True
#
# 训练建议：
#   - loss: class-balanced BCE（你在 losses.py 已实现）
#   - deep supervision: True（trainer 已支持）
#   - pad_stride: 32（trainer/infer_engine 已支持）
#
# =============================================================================

from __future__ import annotations

import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_func


class ConvRelu(nn.Module):
    """
    一个标准的小模块：Conv(3x3) + ReLU
    用于对 backbone 特征做轻度“精炼（refine）”，让 side output 更稳定。

    Args:
        in_ch : 输入通道数
        out_ch: 输出通道数（建议固定为一个较小值，如 32/48/64）
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # 初始化：小随机值（边缘检测中常用做法）
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class RCFWrapper(nn.Module):
    """
    Simplified RCF-style network based on VGG16.

    Backbone:
        torchvision.models.vgg16 (features only)

    Feature taps (after ReLU):
        conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

    For each stage:
        refine: Conv(3x3)->ReLU (reduce channels to mid_ch)
        side  : Conv(1x1)->1 channel
        upsample to input size

    Fusion:
        concat 5 upsampled side logits -> Conv(1x1)->1 channel

    Args:
        pretrained:
            If True, load ImageNet-pretrained VGG16 weights (torchvision will download/cache).
        return_sides_in_train:
            If True and model.training=True, forward returns (fuse_logits, side_logits_list).
            Otherwise returns fuse_logits only.
        mid_ch:
            Channel width after refine blocks. Larger -> potentially stronger but heavier.
            Typical choices: 21 (historical), 32 (good default), 64 (heavier).
    """

    def __init__(self, pretrained: bool = True, return_sides_in_train: bool = True, mid_ch: int = 32):
        super().__init__()
        self.return_sides_in_train = bool(return_sides_in_train)
        self.mid_ch = int(mid_ch)

        # ---------------------------------------------------------------------
        # 1) Build VGG16 backbone
        # ---------------------------------------------------------------------
        try:
            import torchvision

            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            vgg = torchvision.models.vgg16(weights=weights)
        except Exception as e:
            warnings.warn(
                f"[RCF] Failed to create/load VGG16 (pretrained={pretrained}). "
                f"Falling back to random init. Error: {e}",
                RuntimeWarning,
            )
            import torchvision

            vgg = torchvision.models.vgg16(weights=None)

        feats = list(vgg.features.children())

        # VGG16 stage splits（与 HED 一致，便于统一）
        #   stage1: conv1_2 relu (0..3), pool1=4
        #   stage2: conv2_2 relu (5..8), pool2=9
        #   stage3: conv3_3 relu (10..15), pool3=16
        #   stage4: conv4_3 relu (17..22), pool4=23
        #   stage5: conv5_3 relu (24..29)
        self.stage1 = nn.Sequential(*feats[0:4])
        self.pool1 = feats[4]
        self.stage2 = nn.Sequential(*feats[5:9])
        self.pool2 = feats[9]
        self.stage3 = nn.Sequential(*feats[10:16])
        self.pool3 = feats[16]
        self.stage4 = nn.Sequential(*feats[17:23])
        self.pool4 = feats[23]
        self.stage5 = nn.Sequential(*feats[24:30])

        # ---------------------------------------------------------------------
        # 2) Refine blocks (3x3 conv + ReLU) to stabilize side outputs
        #    RCF-style: enrich + compress features before predicting edges.
        # ---------------------------------------------------------------------
        self.ref1 = ConvRelu(64, self.mid_ch)
        self.ref2 = ConvRelu(128, self.mid_ch)
        self.ref3 = ConvRelu(256, self.mid_ch)
        self.ref4 = ConvRelu(512, self.mid_ch)
        self.ref5 = ConvRelu(512, self.mid_ch)

        # ---------------------------------------------------------------------
        # 3) Side output layers: 1x1 conv -> 1 channel logits
        # ---------------------------------------------------------------------
        self.side1 = nn.Conv2d(self.mid_ch, 1, kernel_size=1, bias=True)
        self.side2 = nn.Conv2d(self.mid_ch, 1, kernel_size=1, bias=True)
        self.side3 = nn.Conv2d(self.mid_ch, 1, kernel_size=1, bias=True)
        self.side4 = nn.Conv2d(self.mid_ch, 1, kernel_size=1, bias=True)
        self.side5 = nn.Conv2d(self.mid_ch, 1, kernel_size=1, bias=True)

        # Initialize side convs (small random init)
        for m in (self.side1, self.side2, self.side3, self.side4, self.side5):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0.0)

        # ---------------------------------------------------------------------
        # 4) Fusion layer: concat 5 sides -> 1 channel
        # ---------------------------------------------------------------------
        self.fuse = nn.Conv2d(5, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.fuse.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B,3,H,W) float32
        Returns:
            eval/infer: fuse_logits (B,1,H,W)
            train: (fuse_logits, [s1..s5]) if return_sides_in_train=True
        """
        b, c, h, w = x.shape

        # ---------------------------------------------------------------------
        # 1) Extract multi-level features from VGG16
        # ---------------------------------------------------------------------
        f1 = self.stage1(x)        # (B,  64, H,   W)
        x2 = self.pool1(f1)
        f2 = self.stage2(x2)       # (B, 128, H/2, W/2)
        x3 = self.pool2(f2)
        f3 = self.stage3(x3)       # (B, 256, H/4, W/4)
        x4 = self.pool3(f3)
        f4 = self.stage4(x4)       # (B, 512, H/8, W/8)
        x5 = self.pool4(f4)
        f5 = self.stage5(x5)       # (B, 512, H/16,W/16)

        # ---------------------------------------------------------------------
        # 2) Refine features (compress to mid_ch)
        # ---------------------------------------------------------------------
        r1 = self.ref1(f1)         # (B, mid_ch, H,   W)
        r2 = self.ref2(f2)         # (B, mid_ch, H/2, W/2)
        r3 = self.ref3(f3)         # (B, mid_ch, H/4, W/4)
        r4 = self.ref4(f4)         # (B, mid_ch, H/8, W/8)
        r5 = self.ref5(f5)         # (B, mid_ch, H/16,W/16)

        # ---------------------------------------------------------------------
        # 3) Side logits (1 channel each)
        # ---------------------------------------------------------------------
        s1 = self.side1(r1)        # (B,1,H,   W)
        s2 = self.side2(r2)        # (B,1,H/2, W/2)
        s3 = self.side3(r3)        # (B,1,H/4, W/4)
        s4 = self.side4(r4)        # (B,1,H/8, W/8)
        s5 = self.side5(r5)        # (B,1,H/16,W/16)

        # ---------------------------------------------------------------------
        # 4) Upsample sides to input resolution
        #    align_corners=False: avoids subtle spatial bias and is the common safe choice.
        # ---------------------------------------------------------------------
        s1u = torch_func.interpolate(s1, size=(h, w), mode="bilinear", align_corners=False)
        s2u = torch_func.interpolate(s2, size=(h, w), mode="bilinear", align_corners=False)
        s3u = torch_func.interpolate(s3, size=(h, w), mode="bilinear", align_corners=False)
        s4u = torch_func.interpolate(s4, size=(h, w), mode="bilinear", align_corners=False)
        s5u = torch_func.interpolate(s5, size=(h, w), mode="bilinear", align_corners=False)

        # ---------------------------------------------------------------------
        # 5) Fusion: concat -> 1x1 conv
        # ---------------------------------------------------------------------
        fuse_in = torch.cat([s1u, s2u, s3u, s4u, s5u], dim=1)  # (B,5,H,W)
        fuse_logits = self.fuse(fuse_in)                       # (B,1,H,W)

        # ---------------------------------------------------------------------
        # 6) Output contract (match your trainer/infer_engine)
        # ---------------------------------------------------------------------
        if self.training and self.return_sides_in_train:
            return fuse_logits, [s1u, s2u, s3u, s4u, s5u]
        return fuse_logits