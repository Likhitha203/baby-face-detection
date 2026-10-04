"""Albumentations pipelines for baby vs non-baby face crops."""

from __future__ import annotations

import inspect

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _gauss_noise(p: float = 0.2):
    """Albumentations 2.x: std_range is in [0, 1] on the working image scale. 1.x used var_limit."""
    sig = inspect.signature(A.GaussNoise.__init__)
    if "std_range" in sig.parameters:
        # ~8–28 uint8 stdev ≈ 3–11% of dynamic range when image is float-hwC in [0,1]
        return A.GaussNoise(std_range=(0.03, 0.11), mean_range=(0.0, 0.0), p=p)
    return A.GaussNoise(var_limit=(8.0, 32.0), p=p)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Augment without geometric warping that distorts infant proportions."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=15,
                val_shift_limit=12,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.SmallestMaxSize(max_size=max(image_size + 32, 256)),
            A.RandomCrop(height=image_size, width=image_size, p=0.3),
            A.Resize(height=image_size, width=image_size),
            # Simulate laptop-cam low light / noise
            A.RandomGamma(gamma_limit=(75, 125), p=0.25),
            _gauss_noise(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
