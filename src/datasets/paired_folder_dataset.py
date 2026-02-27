# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
# =============================================================================
# src/datasets/paired_folder_dataset.py
# =============================================================================
#
# 读取“配对的 image / edge”文件夹数据集：
#
# root/
#   dataset/
#     image/{train|val|test}/*.png
#     edge/{train|val|test}/*.png  (同名，GT 边缘：黑底白边，0/255)
#
# 返回：
#   img  : torch.float32 (3,H,W), 值域 [0,1]
#   edge : torch.float32 (1,H,W), 值域 {0,1}
#   meta : dict，包含 filename、原始尺寸等信息（用于保存预测图）
#
# =============================================================================

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedEdgeFolderDataset(Dataset):
    def __init__(self, root_dir: str, dataset_name: str, split: str):
        """
        Args:
            root_dir: 数据根目录（包含多个 dataset 子文件夹）
            dataset_name: 具体数据集名（如 PASD150_0、BIPED）
            split: "train"/"val"/"test"
        """
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split

        self.img_dir = self.root_dir / dataset_name / "image" / split
        self.edge_dir = self.root_dir / dataset_name / "edge" / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not self.edge_dir.exists():
            raise FileNotFoundError(f"Edge dir not found: {self.edge_dir}")

        # 只读取 png（与你当前数据格式一致）
        self.img_paths = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() == ".png"])
        if not self.img_paths:
            raise FileNotFoundError(f"No png images found in: {self.img_dir}")

        # image 与 edge 必须同名配对
        self.edge_paths = []
        for p in self.img_paths:
            ep = self.edge_dir / p.name
            if not ep.exists():
                raise FileNotFoundError(f"Missing edge gt for {p.name}: {ep}")
            self.edge_paths.append(ep)

    def __len__(self) -> int:
        return len(self.img_paths)

    @staticmethod
    def _read_rgb_float01(p: Path) -> torch.Tensor:
        """
        读取 RGB 图像，输出 CHW float32 in [0,1]
        """
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.uint8)  # HWC, 0..255
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        return ten

    @staticmethod
    def _read_edge01(p: Path, thr: int = 128) -> torch.Tensor:
        """
        读取 GT 边缘图，输出 1HW float32 in {0,1}
        - 输入是灰度图（0..255，白色为边缘）
        - thr 以上视为边缘（1）
        """
        edge = Image.open(p).convert("L")
        arr = np.array(edge, dtype=np.uint8)  # HW 0..255
        edge01 = (arr >= thr).astype(np.uint8)  # 0/1
        ten = torch.from_numpy(edge01).unsqueeze(0).contiguous().float()
        return ten

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        edge_path = self.edge_paths[idx]

        img = self._read_rgb_float01(img_path)    # (3,H,W)
        edge = self._read_edge01(edge_path)       # (1,H,W)

        meta = {
            "dataset": self.dataset_name,
            "split": self.split,
            "filename": img_path.name,            # 用于保存预测文件名
            "img_path": str(img_path),
            "edge_path": str(edge_path),
            "h": int(img.shape[1]),
            "w": int(img.shape[2]),
        }
        return img, edge, meta