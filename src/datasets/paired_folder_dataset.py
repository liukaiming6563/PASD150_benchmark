# CREATED BY LKM AND BLESSED BY JMQ
# coding=utf-8
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedEdgeFolderDataset(Dataset):
    """
    读取固定的离线增强数据集目录：

    root/
      dataset/
        image/{train|val|test}/*.png
        edge/{train|val|test}/*.png   (同名，黑底白边GT)

    返回：
      img : (3,H,W) float32, [0,1]
      edge: (1,H,W) float32, {0,1}
      meta: dict (filename/path/size)
    """

    def __init__(self, root_dir: str, dataset_name: str, split: str):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split

        self.img_dir = self.root_dir / dataset_name / "image" / split
        self.edge_dir = self.root_dir / dataset_name / "edge" / split
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not self.edge_dir.exists():
            raise FileNotFoundError(f"Edge dir not found: {self.edge_dir}")

        self.img_paths = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() == ".png"])
        if not self.img_paths:
            raise FileNotFoundError(f"No png images found in: {self.img_dir}")

        self.edge_paths = []
        for p in self.img_paths:
            ep = self.edge_dir / p.name
            if not ep.exists():
                raise FileNotFoundError(f"Missing edge gt for {p.name}: {ep}")
            self.edge_paths.append(ep)

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _read_rgb_float01(p: Path) -> torch.Tensor:
        # RGB png -> float tensor CHW in [0,1]
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.uint8)  # HWC 0..255
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        return ten

    @staticmethod
    def _read_edge01(p: Path, thr: int = 128) -> torch.Tensor:
        # edge png -> float tensor 1HW in {0,1}
        edge = Image.open(p).convert("L")
        arr = np.array(edge, dtype=np.uint8)
        edge01 = (arr >= thr).astype(np.uint8)  # 0/1
        ten = torch.from_numpy(edge01).unsqueeze(0).contiguous().float()
        return ten

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        edge_path = self.edge_paths[idx]

        img = self._read_rgb_float01(img_path)
        edge = self._read_edge01(edge_path)

        meta = {
            "dataset": self.dataset_name,
            "split": self.split,
            "filename": img_path.name,
            "img_path": str(img_path),
            "edge_path": str(edge_path),
            "h": int(img.shape[1]),
            "w": int(img.shape[2]),
        }
        return img, edge, meta