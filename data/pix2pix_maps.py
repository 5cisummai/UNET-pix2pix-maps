from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.BICUBIC)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    return tensor


def _normalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    return image_tensor.mul(2.0).sub(1.0)


class Pix2PixMapsDataset(Dataset[dict[str, Any]]):
    """Loads Pix2Pix paired images where source and target are concatenated horizontally."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        image_size: int = 256,
        load_size: int = 286,
        augment: bool = True,
        source_side: str = "left",
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.load_size = load_size
        self.augment = augment
        self.source_side = source_side.lower()

        if self.source_side not in {"left", "right"}:
            raise ValueError("source_side must be 'left' or 'right'.")

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_paths = sorted(path for path in self.split_dir.iterdir() if path.suffix.lower() in extensions)
        if not self.image_paths:
            raise FileNotFoundError(f"No paired images found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        paired_image = Image.open(image_path).convert("RGB")
        width, height = paired_image.size
        midpoint = width // 2

        left_image = paired_image.crop((0, 0, midpoint, height))
        right_image = paired_image.crop((midpoint, 0, width, height))

        if self.source_side == "left":
            source_image = left_image
            target_image = right_image
        else:
            source_image = right_image
            target_image = left_image

        if self.augment:
            resize_size = (self.load_size, self.load_size)
            source_image = _resize_image(source_image, resize_size)
            target_image = _resize_image(target_image, resize_size)

            crop_limit = self.load_size - self.image_size
            top = int(torch.randint(0, crop_limit + 1, (1,)).item())
            left = int(torch.randint(0, crop_limit + 1, (1,)).item())
            crop_box = (left, top, left + self.image_size, top + self.image_size)

            source_image = source_image.crop(crop_box)
            target_image = target_image.crop(crop_box)

            if torch.rand(1).item() > 0.5:
                source_image = source_image.transpose(Image.FLIP_LEFT_RIGHT)
                target_image = target_image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            resize_size = (self.image_size, self.image_size)
            source_image = _resize_image(source_image, resize_size)
            target_image = _resize_image(target_image, resize_size)

        source_tensor = _normalize_image(_to_tensor(source_image))
        target_tensor = _normalize_image(_to_tensor(target_image))

        return {
            "source": source_tensor,
            "target": target_tensor,
            "path": str(image_path),
        }
