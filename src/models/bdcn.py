# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/models/bdcn.py
# =============================================================================
#
# BDCN (Bi-Directional Cascade Network) - Simplified BDCN-style Baseline
# -----------------------------------------------------------------------------
# 重要说明（与你论文表述一致）：
#   本实现是 “BDCN-style baseline / BDCN variant”，用于统一 benchmark。
#   它保留 BDCN 的关键思想（级联式多尺度融合 + 深监督）：
#     1) VGG16 多层特征（multi-level features）
#     2) 每个尺度都有 side output，并进行 deep supervision
#     3) “级联（cascade）”：更深层的预测会显式融合更浅层/前一层的预测信息
#
# 简化点：
#   - 严格 BDCN 论文包含更复杂的双向级联（bi-directional）和更多细节。
#   - 为了保证 5 datasets × 6 models 的公平比较与可维护性，
#     我们实现一个稳定、通用、可复现的 BDCN-style 级联结构：
#       - 自上而下（deep->shallow）和自下而上（shallow->deep）两条“简化级联”分支
#       - 每条分支都采用：concat(当前尺度特征, 相邻尺度的预测logits) -> 1x1 conv -> logits
#       - 最终融合 fuse = Conv1x1(concat(all cascaded logits))
#
# 输入/输出约定（必须与 engine 对齐）：
#   输入 x : (B,3,H,W) float32 in [0,1]
#   输出 logits（未 sigmoid）：
#       eval/infer: fuse_logits (B,1,H,W)
#       train: (fuse_logits, side_logits_list) if return_sides_in_train=True
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
    轻量 refine block：Conv(3x3) + ReLU
    用途：对 backbone 特征做“通道压缩 + 局部精炼”，让边缘预测更稳定。

    Args:
        in_ch : 输入通道数
        out_ch: 输出通道数（建议 32/48/64，越大越强但更耗显存）
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # 边缘检测里常用：side/refine 小随机初始化
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class BDCNWrapper(nn.Module):
    """
    Simplified BDCN-style cascade network (VGG16 backbone).

    Backbone taps (after ReLU):
      conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

    Notation:
      - f1..f5: VGG stage features (channels: 64,128,256,512,512)
      - r1..r5: refined features (channels: mid_ch)
      - td: top-down cascaded logits (deep -> shallow)
      - bu: bottom-up cascaded logits (shallow -> deep)
      - fuse: fusion of all cascaded logits

    Why cascade?
      - BDCN 核心是“跨尺度的级联监督/融合”。
      - 我们用一种稳定、易实现的方式体现 cascade：
          当前尺度的输出 = Conv1x1(concat(refined_feature, neighbor_logit))
        其中 neighbor_logit 来自相邻尺度（上一级或下一级）的预测。
      - 这样可以显式把“高层语义预测”与“低层细节预测”互相传递。

    Args:
        pretrained: 是否加载 ImageNet 预训练 VGG16
        return_sides_in_train: train 时是否返回 side logits list
        mid_ch: refine 后的通道数（默认 32）
    """

    def __init__(self, pretrained: bool = True, return_sides_in_train: bool = True, mid_ch: int = 32):
        super().__init__()
        self.return_sides_in_train = bool(return_sides_in_train)
        self.mid_ch = int(mid_ch)

        # ---------------------------------------------------------------------
        # 1) VGG16 backbone
        # ---------------------------------------------------------------------
        try:
            import torchvision

            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            vgg = torchvision.models.vgg16(weights=weights)
        except Exception as e:
            warnings.warn(
                f"[BDCN] Failed to create/load VGG16 (pretrained={pretrained}). "
                f"Falling back to random init. Error: {e}",
                RuntimeWarning,
            )
            import torchvision

            vgg = torchvision.models.vgg16(weights=None)

        feats = list(vgg.features.children())

        # 与 HED/RCF 同样的 stage 切分，便于统一
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
        # 2) Refine blocks：把不同尺度通道统一压到 mid_ch
        # ---------------------------------------------------------------------
        self.ref1 = ConvRelu(64, self.mid_ch)
        self.ref2 = ConvRelu(128, self.mid_ch)
        self.ref3 = ConvRelu(256, self.mid_ch)
        self.ref4 = ConvRelu(512, self.mid_ch)
        self.ref5 = ConvRelu(512, self.mid_ch)

        # ---------------------------------------------------------------------
        # 3) Cascade heads
        #
        # 我们做两条简化级联：
        #   (A) Top-Down：从深层到浅层传递预测
        #       td5 = head_td5(r5)
        #       td4 = head_td4(concat(r4, up(td5)))
        #       td3 = head_td3(concat(r3, up(td4)))
        #       td2 = head_td2(concat(r2, up(td3)))
        #       td1 = head_td1(concat(r1, up(td2)))
        #
        #   (B) Bottom-Up：从浅层到深层传递预测
        #       bu1 = head_bu1(r1)
        #       bu2 = head_bu2(concat(r2, down(bu1)))
        #       bu3 = head_bu3(concat(r3, down(bu2)))
        #       bu4 = head_bu4(concat(r4, down(bu3)))
        #       bu5 = head_bu5(concat(r5, down(bu4)))
        #
        # 每个 head 都是 1x1 conv，把 concat 后的特征映射到 1 通道 logits。
        # ---------------------------------------------------------------------
        # top-down heads
        self.head_td5 = nn.Conv2d(self.mid_ch, 1, kernel_size=1, bias=True)
        self.head_td4 = nn.Conv2d(self.mid_ch + 1, 1, kernel_size=1, bias=True)
        self.head_td3 = nn.Conv2d(self.mid_ch + 1, 1, kernel_size=1, bias=True)
        self.head_td2 = nn.Conv2d(self.mid_ch + 1, 1, kernel_size=1, bias=True)
        self.head_td1 = nn.Conv2d(self.mid_ch + 1, 1, kernel_size=1, bias=True)

        # bottom-up heads
        self.head_bu1 = nn.Conv2d(self.mid_ch, 1, kernel_size=1, bias=True)
        self.head_bu2 = nn.Conv2d(self.mid_ch + 1, 1, kernel_size=1, bias=True)
        self.head_bu3 = nn.Conv2d(self.mid_ch + 1, 1, kernel_size=1, bias=True)
        self.head_bu4 = nn.Conv2d(self.mid_ch + 1, 1, kernel_size=1, bias=True)
        self.head_bu5 = nn.Conv2d(self.mid_ch + 1, 1, kernel_size=1, bias=True)

        # 初始化 cascade heads（小随机 init）
        for m in (
            self.head_td5, self.head_td4, self.head_td3, self.head_td2, self.head_td1,
            self.head_bu1, self.head_bu2, self.head_bu3, self.head_bu4, self.head_bu5
        ):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0.0)

        # ---------------------------------------------------------------------
        # 4) Fusion：融合所有级联 logits（共 10 张：td1..td5 + bu1..bu5）
        # ---------------------------------------------------------------------
        self.fuse = nn.Conv2d(10, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.fuse.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fuse.bias, 0.0)

    @staticmethod
    def _upsample_to(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        """
        双线性上采样到目标尺寸。
        align_corners=False：一般更稳定，避免空间偏移。
        """
        return torch_func.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _downsample_to(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        """
        双线性下采样到目标尺寸（用于 bottom-up 分支把浅层 logits 传到深层尺度）。
        """
        return torch_func.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B,3,H,W) float32 in [0,1]
        Returns:
            eval: fuse_logits (B,1,H,W)
            train: (fuse_logits, side_logits_list)  # deep supervision
        """
        b, c, H, W = x.shape

        # ---------------------------------------------------------------------
        # 1) VGG multi-level features
        # ---------------------------------------------------------------------
        f1 = self.stage1(x)             # (B,64, H,   W)
        x2 = self.pool1(f1)
        f2 = self.stage2(x2)            # (B,128,H/2, W/2)
        x3 = self.pool2(f2)
        f3 = self.stage3(x3)            # (B,256,H/4, W/4)
        x4 = self.pool3(f3)
        f4 = self.stage4(x4)            # (B,512,H/8, W/8)
        x5 = self.pool4(f4)
        f5 = self.stage5(x5)            # (B,512,H/16,W/16)

        # 尺寸记录（用于 up/downsample）
        s1_hw = (f1.shape[2], f1.shape[3])
        s2_hw = (f2.shape[2], f2.shape[3])
        s3_hw = (f3.shape[2], f3.shape[3])
        s4_hw = (f4.shape[2], f4.shape[3])
        s5_hw = (f5.shape[2], f5.shape[3])

        # ---------------------------------------------------------------------
        # 2) Refine
        # ---------------------------------------------------------------------
        r1 = self.ref1(f1)              # (B,mid,H,   W)
        r2 = self.ref2(f2)              # (B,mid,H/2, W/2)
        r3 = self.ref3(f3)              # (B,mid,H/4, W/4)
        r4 = self.ref4(f4)              # (B,mid,H/8, W/8)
        r5 = self.ref5(f5)              # (B,mid,H/16,W/16)

        # ---------------------------------------------------------------------
        # 3A) Top-Down cascade (deep -> shallow)
        # ---------------------------------------------------------------------
        td5 = self.head_td5(r5)  # (B,1,H/16,W/16)

        td5_up4 = self._upsample_to(td5, s4_hw)
        td4 = self.head_td4(torch.cat([r4, td5_up4], dim=1))  # (B,1,H/8,W/8)

        td4_up3 = self._upsample_to(td4, s3_hw)
        td3 = self.head_td3(torch.cat([r3, td4_up3], dim=1))  # (B,1,H/4,W/4)

        td3_up2 = self._upsample_to(td3, s2_hw)
        td2 = self.head_td2(torch.cat([r2, td3_up2], dim=1))  # (B,1,H/2,W/2)

        td2_up1 = self._upsample_to(td2, s1_hw)
        td1 = self.head_td1(torch.cat([r1, td2_up1], dim=1))  # (B,1,H,W)

        # ---------------------------------------------------------------------
        # 3B) Bottom-Up cascade (shallow -> deep)
        # ---------------------------------------------------------------------
        bu1 = self.head_bu1(r1)  # (B,1,H,W)

        bu1_down2 = self._downsample_to(bu1, s2_hw)
        bu2 = self.head_bu2(torch.cat([r2, bu1_down2], dim=1))  # (B,1,H/2,W/2)

        bu2_down3 = self._downsample_to(bu2, s3_hw)
        bu3 = self.head_bu3(torch.cat([r3, bu2_down3], dim=1))  # (B,1,H/4,W/4)

        bu3_down4 = self._downsample_to(bu3, s4_hw)
        bu4 = self.head_bu4(torch.cat([r4, bu3_down4], dim=1))  # (B,1,H/8,W/8)

        bu4_down5 = self._downsample_to(bu4, s5_hw)
        bu5 = self.head_bu5(torch.cat([r5, bu4_down5], dim=1))  # (B,1,H/16,W/16)

        # ---------------------------------------------------------------------
        # 4) Upsample all cascaded logits to input resolution (H,W)
        # ---------------------------------------------------------------------
        td1u = self._upsample_to(td1, (H, W))  # td1 本来就是 (H,W)，这里写统一形式
        td2u = self._upsample_to(td2, (H, W))
        td3u = self._upsample_to(td3, (H, W))
        td4u = self._upsample_to(td4, (H, W))
        td5u = self._upsample_to(td5, (H, W))

        bu1u = self._upsample_to(bu1, (H, W))  # bu1 本来就是 (H,W)
        bu2u = self._upsample_to(bu2, (H, W))
        bu3u = self._upsample_to(bu3, (H, W))
        bu4u = self._upsample_to(bu4, (H, W))
        bu5u = self._upsample_to(bu5, (H, W))

        # ---------------------------------------------------------------------
        # 5) Fusion: concat 10 maps -> fuse logits
        # ---------------------------------------------------------------------
        fuse_in = torch.cat([td1u, td2u, td3u, td4u, td5u, bu1u, bu2u, bu3u, bu4u, bu5u], dim=1)  # (B,10,H,W)
        fuse_logits = self.fuse(fuse_in)  # (B,1,H,W)

        # ---------------------------------------------------------------------
        # 6) Output contract (match your trainer/infer_engine)
        # ---------------------------------------------------------------------
        if self.training and self.return_sides_in_train:
            # side logits list：用于 deep supervision
            # 你可以只监督一条分支，也可以监督全部（这里默认全部，更强但计算稍多）
            side_logits_list: List[torch.Tensor] = [td1u, td2u, td3u, td4u, td5u, bu1u, bu2u, bu3u, bu4u, bu5u]
            return fuse_logits, side_logits_list

        return fuse_logits