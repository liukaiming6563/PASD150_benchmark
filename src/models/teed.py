# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/models/teed.py
# =============================================================================
#
# TEED - Tiny / Efficient Edge Detector (TEED-style baseline/variant)
# -----------------------------------------------------------------------------
# 重要说明（与你论文表述一致）：
#   由于“TEED”在不同工作/仓库中可能指代不同实现，本文件提供一个
#   “TEED-style lightweight baseline/variant”，用于统一 benchmark 的轻量模型对比。
#
# 设计目标：
#   1) 轻量、快速：MobileNetV2-like backbone (Inverted Residual blocks)
#   2) 边缘检测常用范式：多尺度 side outputs + fusion + deep supervision
#   3) 与你现有训练/推理引擎完全兼容：
#        - forward 输出 logits（未 sigmoid）
#        - eval: fuse_logits (B,1,H,W)
#        - train: (fuse_logits, side_logits_list)
#
# 输入/输出约定：
#   输入 x : (B,3,H,W) float32 in [0,1]
#   输出 logits（未 sigmoid）：
#       eval/infer: fuse_logits (B,1,H,W)
#       train: (fuse_logits, side_logits_list)
#
# 训练建议：
#   - loss: class-balanced BCE（你在 losses.py 已实现）
#   - deep supervision: True（trainer 已支持）
#   - pad_stride: 32（trainer/infer_engine 已支持）
#
# =============================================================================

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_func


# =============================================================================
# 1) 基础模块：Conv-BN-ReLU6（MobileNet 系列常用）
# =============================================================================
class ConvBNAct(nn.Module):
    """
    Conv2d + BatchNorm + Activation

    Args:
        in_ch/out_ch: 输入/输出通道
        k, s, p: kernel/stride/padding
        groups: groups=out_ch 代表 depthwise conv
        act: 激活函数（默认 ReLU6，符合 MobileNetV2 常用配置）
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        groups: int = 1,
        act: nn.Module | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act if act is not None else nn.ReLU6(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# =============================================================================
# 2) Inverted Residual Block（MobileNetV2 核心）
# =============================================================================
class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual block.

    Structure:
      - (optional) expand: 1x1 conv  (in_ch -> in_ch*expand)
      - depthwise: 3x3 dw conv
      - project: 1x1 conv (to out_ch)
      - residual if (stride==1 and in_ch==out_ch)

    Args:
        in_ch/out_ch: 输入/输出通道
        stride: 1 或 2（stride=2 用于下采样）
        expand_ratio: expansion factor（常用 2~6）
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(round(in_ch * expand_ratio))

        self.use_res = (stride == 1 and in_ch == out_ch)

        layers: List[nn.Module] = []

        # expand
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden, k=1, s=1, p=0, act=nn.ReLU6(inplace=True)))

        # depthwise
        layers.append(ConvBNAct(hidden, hidden, k=3, s=stride, p=1, groups=hidden, act=nn.ReLU6(inplace=True)))

        # project (linear)
        proj = nn.Conv2d(hidden, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(proj.weight, mode="fan_out", nonlinearity="relu")
        layers.append(proj)
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            return x + y
        return y


# =============================================================================
# 3) TEED-style 网络：轻量骨架 + 多尺度 side outputs + fuse
# =============================================================================
class TEEDWrapper(nn.Module):
    """
    TEED-style lightweight edge detector.

    Backbone design (MobileNetV2-like):
      - stem: stride=2 (H/2)
      - stage1: stride=1
      - stage2: stride=2 (H/4)
      - stage3: stride=2 (H/8)
      - stage4: stride=2 (H/16)
      - stage5: stride=1 (keep H/16, deeper features)

    We tap 5 multi-scale features: f1..f5
      - f1: H/2
      - f2: H/4
      - f3: H/8
      - f4: H/16
      - f5: H/16 (deeper)

    Side outputs:
      - Each feature -> 1x1 conv -> 1 channel logit
      - Upsample to input size (H,W)
    Fusion:
      - concat 5 upsampled logits -> 1x1 conv -> fuse logit

    Args:
        return_sides_in_train: train 时返回 side logits list（用于 deep supervision）
        width_mult: 通道宽度倍率（越小越轻量）
            - 0.5: 更轻量
            - 1.0: 默认
        expand_ratio: inverted residual expansion ratio（默认 6，MobileNetV2 常用）
    """

    def __init__(
        self,
        return_sides_in_train: bool = True,
        width_mult: float = 1.0,
        expand_ratio: int = 6,
    ):
        super().__init__()
        self.return_sides_in_train = bool(return_sides_in_train)

        def c(ch: int) -> int:
            """按 width_mult 缩放通道数，至少为 8（避免太小导致 BN 不稳定）"""
            return max(8, int(round(ch * width_mult)))

        # ---------------------------------------------------------------------
        # 1) Stem: 3 -> c(16), stride=2 => H/2
        # ---------------------------------------------------------------------
        self.stem = ConvBNAct(3, c(16), k=3, s=2, p=1, act=nn.ReLU6(inplace=True))

        # ---------------------------------------------------------------------
        # 2) Backbone stages (MobileNetV2-like)
        #    输出多尺度特征 f1..f5
        # ---------------------------------------------------------------------
        # stage1 (H/2)
        self.stage1 = nn.Sequential(
            InvertedResidual(c(16), c(16), stride=1, expand_ratio=1),
            InvertedResidual(c(16), c(16), stride=1, expand_ratio=expand_ratio),
        )
        # stage2 (H/4)
        self.stage2 = nn.Sequential(
            InvertedResidual(c(16), c(24), stride=2, expand_ratio=expand_ratio),
            InvertedResidual(c(24), c(24), stride=1, expand_ratio=expand_ratio),
        )
        # stage3 (H/8)
        self.stage3 = nn.Sequential(
            InvertedResidual(c(24), c(32), stride=2, expand_ratio=expand_ratio),
            InvertedResidual(c(32), c(32), stride=1, expand_ratio=expand_ratio),
        )
        # stage4 (H/16)
        self.stage4 = nn.Sequential(
            InvertedResidual(c(32), c(64), stride=2, expand_ratio=expand_ratio),
            InvertedResidual(c(64), c(64), stride=1, expand_ratio=expand_ratio),
        )
        # stage5 (H/16, deeper)
        self.stage5 = nn.Sequential(
            InvertedResidual(c(64), c(96), stride=1, expand_ratio=expand_ratio),
            InvertedResidual(c(96), c(96), stride=1, expand_ratio=expand_ratio),
        )

        # ---------------------------------------------------------------------
        # 3) Side heads: 1x1 conv -> 1 channel logits
        # ---------------------------------------------------------------------
        self.side1 = nn.Conv2d(c(16), 1, kernel_size=1, bias=True)
        self.side2 = nn.Conv2d(c(24), 1, kernel_size=1, bias=True)
        self.side3 = nn.Conv2d(c(32), 1, kernel_size=1, bias=True)
        self.side4 = nn.Conv2d(c(64), 1, kernel_size=1, bias=True)
        self.side5 = nn.Conv2d(c(96), 1, kernel_size=1, bias=True)

        for m in (self.side1, self.side2, self.side3, self.side4, self.side5):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0.0)

        # ---------------------------------------------------------------------
        # 4) Fuse: concat 5 side logits -> 1 channel
        # ---------------------------------------------------------------------
        self.fuse = nn.Conv2d(5, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.fuse.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B,3,H,W)
        Returns:
            eval/infer: fuse_logits (B,1,H,W)
            train: (fuse_logits, [s1..s5]) if return_sides_in_train=True
        """
        b, c, H, W = x.shape

        # Stem
        x = self.stem(x)      # (B,c16,H/2,W/2)

        # Multi-scale backbone features
        f1 = self.stage1(x)   # (B,c16,H/2,W/2)
        f2 = self.stage2(f1)  # (B,c24,H/4,W/4)
        f3 = self.stage3(f2)  # (B,c32,H/8,W/8)
        f4 = self.stage4(f3)  # (B,c64,H/16,W/16)
        f5 = self.stage5(f4)  # (B,c96,H/16,W/16)

        # Side logits
        s1 = self.side1(f1)   # (B,1,H/2,W/2)
        s2 = self.side2(f2)   # (B,1,H/4,W/4)
        s3 = self.side3(f3)   # (B,1,H/8,W/8)
        s4 = self.side4(f4)   # (B,1,H/16,W/16)
        s5 = self.side5(f5)   # (B,1,H/16,W/16)

        # Upsample all to input size
        s1u = torch_func.interpolate(s1, size=(H, W), mode="bilinear", align_corners=False)
        s2u = torch_func.interpolate(s2, size=(H, W), mode="bilinear", align_corners=False)
        s3u = torch_func.interpolate(s3, size=(H, W), mode="bilinear", align_corners=False)
        s4u = torch_func.interpolate(s4, size=(H, W), mode="bilinear", align_corners=False)
        s5u = torch_func.interpolate(s5, size=(H, W), mode="bilinear", align_corners=False)

        fuse_in = torch.cat([s1u, s2u, s3u, s4u, s5u], dim=1)  # (B,5,H,W)
        fuse_logits = self.fuse(fuse_in)                        # (B,1,H,W)

        if self.training and self.return_sides_in_train:
            return fuse_logits, [s1u, s2u, s3u, s4u, s5u]

        return fuse_logits