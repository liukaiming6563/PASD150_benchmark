# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/models/pidinet.py
# =============================================================================
#
# PiDiNet (Pixel Difference Network) - PiDiNet-style lightweight edge detector
# -----------------------------------------------------------------------------
# 重要说明（与你论文表述一致）：
#   本实现是 “PiDiNet-style baseline / PiDiNet variant”，用于统一 benchmark。
#   它保留 PiDiNet 的核心思想：
#     1) 轻量 backbone（适合速度/参数量对比）
#     2) Pixel Difference Convolution（PDC）的“中心差分”形式（CD-PDC），增强边缘敏感性
#     3) 多尺度 side outputs + fusion + deep supervision（边缘检测主流训练范式）
#
# 关于 PDC（Central Difference, CD）：
#   CD-PDC 的直觉：
#       输出更关注邻域像素与中心像素的差异（对边缘/梯度更敏感）。
#   数学上可写成：
#       y = sum_{k} w_k * x_k  -  theta * x_center * sum_{k} w_k
#   等价为一个“修改后的卷积核”：
#       w'_k = w_k               (k != center)
#       w'_center = w_center - theta * sum_{k} w_k
#   其中 theta 常取 1（或 0.7/0.9），本实现默认 theta=1。
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

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_func


# =============================================================================
# 1) CD-PDC: Central Difference Convolution (lightweight)
# =============================================================================
class CDPDCConv2d(nn.Module):
    """
    Central Difference Pixel Difference Convolution (CD-PDC).

    本模块内部维护一个普通的 Conv2d 权重 W，但 forward 时会构造“差分版卷积核” W'：
        W'_k = W_k                               (k != center)
        W'_center = W_center - theta * sum(W_k)  (sum over all k in kernel)

    这样输出会显式减去中心像素项的加权和，增强边缘/梯度响应。

    Args:
        in_ch, out_ch: 输入/输出通道
        kernel_size: 仅实现 3（PiDiNet 中最常用）
        stride, padding, dilation, groups: 与 Conv2d 同义
        bias: 是否使用 bias
        theta: 中心差分强度（通常 1.0 左右）
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        theta: float = 1.0,
    ):
        super().__init__()
        assert kernel_size == 3, "This simplified CD-PDC only supports 3x3 kernels."
        self.theta = float(theta)

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # 初始化建议：用 kaiming 更稳（轻量网络常用）
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,Cin,H,W)
        Returns:
            y: (B,Cout,H',W')
        """
        w = self.conv.weight  # (Cout, Cin/groups, 3, 3)

        # sum over spatial kernel positions: (Cout, Cin/groups)
        w_sum = w.sum(dim=(2, 3), keepdim=False)

        # 构造 w' = w，然后只修改中心位置 (1,1)
        w_cd = w.clone()
        w_cd[:, :, 1, 1] = w_cd[:, :, 1, 1] - self.theta * w_sum

        return torch_func.conv2d(
            x,
            w_cd,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


# =============================================================================
# 2) Lightweight blocks (Depthwise separable + CD-PDC pointwise)
# =============================================================================
class DWConvBNReLU(nn.Module):
    """
    Depthwise Conv + BN + ReLU
    - depthwise: groups = in_ch
    """

    def __init__(self, ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, stride=stride, padding=1, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.dw.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.dw(x)))


class PDCPointwiseBNReLU(nn.Module):
    """
    1x1 pointwise conv 用 CD-PDC 的思想意义不大，
    所以这里用普通 1x1 conv（更快）+ BN + ReLU。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.pw.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(x)))


class PDCBlock(nn.Module):
    """
    PiDiNet-style block (lightweight):
      DWConv(3x3) -> BN -> ReLU
      CD-PDCConv(3x3) -> BN -> ReLU   (用于强化边缘敏感性)
      PWConv(1x1) -> BN -> ReLU

    说明：
      - 这里把 CD-PDC 放在中间的 3x3 中使用
      - 通过 stride 控制下采样（构建多尺度特征）
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, theta: float = 1.0):
        super().__init__()
        # 先把通道变成 out_ch（用 pointwise），再做 depthwise / pdc
        self.proj = PDCPointwiseBNReLU(in_ch, out_ch)

        self.dw = DWConvBNReLU(out_ch, stride=stride)

        self.pdc = CDPDCConv2d(
            in_ch=out_ch,
            out_ch=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            theta=theta,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.pw = PDCPointwiseBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.dw(x)
        x = self.act(self.bn(self.pdc(x)))
        x = self.pw(x)
        return x


# =============================================================================
# 3) PiDiNet-style network with 5 side outputs + fuse
# =============================================================================
class PiDiNetWrapper(nn.Module):
    """
    PiDiNet-style lightweight edge detector.

    Structure:
      stem -> stage1 -> stage2 -> stage3 -> stage4 -> stage5
    Each stage provides a side output (1 channel), upsampled to input size.
    Final output = fuse(concat(side1..side5))

    Args:
        return_sides_in_train:
            Train 时是否返回 side logits list（用于 deep supervision）
        channels:
            每个 stage 的通道数（越大越强但越慢）
        theta:
            CD-PDC 的差分强度
    """

    def __init__(
        self,
        return_sides_in_train: bool = True,
        channels: Tuple[int, int, int, int, int] = (16, 32, 64, 96, 128),
        theta: float = 1.0,
    ):
        super().__init__()
        self.return_sides_in_train = bool(return_sides_in_train)
        c1, c2, c3, c4, c5 = channels

        # ---------------------------------------------------------------------
        # Stem: 3 -> c1
        # ---------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.stem[0].weight, mode="fan_out", nonlinearity="relu")

        # ---------------------------------------------------------------------
        # Stages: progressively downsample to build multi-scale features
        # Strides: 1,2,2,2,2 -> total stride 16 at stage5
        # ---------------------------------------------------------------------
        self.stage1 = nn.Sequential(
            PDCBlock(c1, c1, stride=1, theta=theta),
            PDCBlock(c1, c1, stride=1, theta=theta),
        )
        self.stage2 = nn.Sequential(
            PDCBlock(c1, c2, stride=2, theta=theta),
            PDCBlock(c2, c2, stride=1, theta=theta),
        )
        self.stage3 = nn.Sequential(
            PDCBlock(c2, c3, stride=2, theta=theta),
            PDCBlock(c3, c3, stride=1, theta=theta),
        )
        self.stage4 = nn.Sequential(
            PDCBlock(c3, c4, stride=2, theta=theta),
            PDCBlock(c4, c4, stride=1, theta=theta),
        )
        self.stage5 = nn.Sequential(
            PDCBlock(c4, c5, stride=2, theta=theta),
            PDCBlock(c5, c5, stride=1, theta=theta),
        )

        # ---------------------------------------------------------------------
        # Side heads: 1x1 conv -> 1 channel logits
        # ---------------------------------------------------------------------
        self.side1 = nn.Conv2d(c1, 1, kernel_size=1, bias=True)
        self.side2 = nn.Conv2d(c2, 1, kernel_size=1, bias=True)
        self.side3 = nn.Conv2d(c3, 1, kernel_size=1, bias=True)
        self.side4 = nn.Conv2d(c4, 1, kernel_size=1, bias=True)
        self.side5 = nn.Conv2d(c5, 1, kernel_size=1, bias=True)

        for m in (self.side1, self.side2, self.side3, self.side4, self.side5):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0.0)

        # Fuse: concat 5 side logits -> 1 channel
        self.fuse = nn.Conv2d(5, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.fuse.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B,3,H,W)
        Returns:
            eval: fuse_logits (B,1,H,W)
            train: (fuse_logits, [s1..s5]) if return_sides_in_train=True
        """
        b, c, H, W = x.shape

        # Multi-scale features
        x = self.stem(x)     # (B,c1,H,W)
        f1 = self.stage1(x)  # (B,c1,H,W)
        f2 = self.stage2(f1) # (B,c2,H/2,W/2)
        f3 = self.stage3(f2) # (B,c3,H/4,W/4)
        f4 = self.stage4(f3) # (B,c4,H/8,W/8)
        f5 = self.stage5(f4) # (B,c5,H/16,W/16)

        # Side logits
        s1 = self.side1(f1)
        s2 = self.side2(f2)
        s3 = self.side3(f3)
        s4 = self.side4(f4)
        s5 = self.side5(f5)

        # Upsample to input size
        s1u = torch_func.interpolate(s1, size=(H, W), mode="bilinear", align_corners=False)
        s2u = torch_func.interpolate(s2, size=(H, W), mode="bilinear", align_corners=False)
        s3u = torch_func.interpolate(s3, size=(H, W), mode="bilinear", align_corners=False)
        s4u = torch_func.interpolate(s4, size=(H, W), mode="bilinear", align_corners=False)
        s5u = torch_func.interpolate(s5, size=(H, W), mode="bilinear", align_corners=False)

        # Fuse
        fuse_in = torch.cat([s1u, s2u, s3u, s4u, s5u], dim=1)  # (B,5,H,W)
        fuse_logits = self.fuse(fuse_in)                       # (B,1,H,W)

        if self.training and self.return_sides_in_train:
            return fuse_logits, [s1u, s2u, s3u, s4u, s5u]
        return fuse_logits