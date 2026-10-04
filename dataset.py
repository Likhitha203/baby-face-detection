"""PyTorch Dataset for face crops under data/baby and data/non_baby."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _collect_images(root: Path, exts: frozenset[str]) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


class BabyFaceDataset(Dataset):
    """
    Label 0 = baby, 1 = non_baby (matches logits index for CrossEntropyLoss).
    Expects:
        root/baby/*.jpg|png|...
        root/non_baby/*.jpg|png|...
    """

    EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        baby_subdir: str = "baby",
        non_baby_subdir: str = "non_baby",
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        baby_dir = self.root / baby_subdir
        non_dir = self.root / non_baby_subdir

        baby_paths = _collect_images(baby_dir, self.EXTENSIONS)
        non_paths = _collect_images(non_dir, self.EXTENSIONS)

        self.samples: list[tuple[Path, int]] = [(p, 0) for p in baby_paths] + [
            (p, 1) for p in non_paths
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)
        if self.transform is not None:
            augmented = self.transform(image=image_np)
            tensor = augmented["image"]
        else:
            tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        return tensor, label
