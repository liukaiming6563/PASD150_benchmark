# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/models/hed.py
# =============================================================================
#
# HED（Holistically-Nested Edge Detection）主流实现：
#   - VGG16 backbone（ImageNet 预训练）
#   - 5 个 side outputs + fuse
#   - deep supervision（训练时可返回 side outputs 用于额外 loss）
#
# forward 输出：
#   - eval/infer：fuse_logits (B,1,H,W)  (未 sigmoid)
#   - train（return_sides_in_train=True）：(fuse_logits, [side1..side5])
#
# 注意：
#   - 本模型输出 logits，不做 sigmoid；sigmoid 在 infer_engine / loss 内部处理更灵活
#
# =============================================================================

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as torch_func


class HEDWrapper(nn.Module):
    def __init__(self, pretrained: bool = True, return_sides_in_train: bool = True):
        super().__init__()
        self.return_sides_in_train = bool(return_sides_in_train)

        # torchvision VGG16
        try:
            import torchvision
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            vgg = torchvision.models.vgg16(weights=weights)
        except Exception as e:
            warnings.warn(
                f"[HED] Failed to load pretrained VGG16, fall back to random init. Error: {e}",
                RuntimeWarning,
            )
            import torchvision
            vgg = torchvision.models.vgg16(weights=None)

        feats = list(vgg.features.children())

        # VGG stage split（与经典 HED 对齐）
        self.stage1 = nn.Sequential(*feats[0:4])   # conv1_2 relu
        self.pool1 = feats[4]
        self.stage2 = nn.Sequential(*feats[5:9])   # conv2_2 relu
        self.pool2 = feats[9]
        self.stage3 = nn.Sequential(*feats[10:16]) # conv3_3 relu
        self.pool3 = feats[16]
        self.stage4 = nn.Sequential(*feats[17:23]) # conv4_3 relu
        self.pool4 = feats[23]
        self.stage5 = nn.Sequential(*feats[24:30]) # conv5_3 relu

        # side 1x1 conv
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)

        # fuse
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

        self._init_side_fuse()

    def _init_side_fuse(self) -> None:
        # side/fuse 小随机初始化（常见做法）
        for m in (self.side1, self.side2, self.side3, self.side4, self.side5, self.fuse):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B,3,H,W) float32
        Returns:
            eval: fuse_logits (B,1,H,W)
            train: (fuse_logits, [s1..s5]) if return_sides_in_train True
        """
        _, _, h, w = x.shape

        f1 = self.stage1(x)      # (B,64,H,W)
        x2 = self.pool1(f1)
        f2 = self.stage2(x2)     # (B,128,H/2,W/2)
        x3 = self.pool2(f2)
        f3 = self.stage3(x3)     # (B,256,H/4,W/4)
        x4 = self.pool3(f3)
        f4 = self.stage4(x4)     # (B,512,H/8,W/8)
        x5 = self.pool4(f4)
        f5 = self.stage5(x5)     # (B,512,H/16,W/16)

        s1 = self.side1(f1)
        s2 = self.side2(f2)
        s3 = self.side3(f3)
        s4 = self.side4(f4)
        s5 = self.side5(f5)

        # 上采样到输入大小（align_corners=False 更稳定，避免对齐偏差）
        s1u = torch_func.interpolate(s1, size=(h, w), mode="bilinear", align_corners=False)
        s2u = torch_func.interpolate(s2, size=(h, w), mode="bilinear", align_corners=False)
        s3u = torch_func.interpolate(s3, size=(h, w), mode="bilinear", align_corners=False)
        s4u = torch_func.interpolate(s4, size=(h, w), mode="bilinear", align_corners=False)
        s5u = torch_func.interpolate(s5, size=(h, w), mode="bilinear", align_corners=False)

        fuse_in = torch.cat([s1u, s2u, s3u, s4u, s5u], dim=1)  # (B,5,H,W)
        fuse_logits = self.fuse(fuse_in)                        # (B,1,H,W)

        if self.training and self.return_sides_in_train:
            return fuse_logits, [s1u, s2u, s3u, s4u, s5u]
        return fuse_logits