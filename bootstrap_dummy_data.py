#!/usr/bin/env python3
"""
Create tiny synthetic JPG crops so `train.py` can run (pipeline smoke test only).

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_random_face_crop(dest: Path, seed: int, size: tuple[int, int]) -> None:
    rng = np.random.default_rng(seed)
    h, w = size
    # Slight color bias per "class" so the network has trivial signal (still not real faces).
    base = rng.integers(40, 80, size=(h, w, 3), dtype=np.uint8)
    noise = rng.integers(-25, 25, size=(h, w, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(img).save(dest, format="JPEG", quality=92)
    except ImportError:
        import cv2

        cv2.imwrite(str(dest), img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--n-baby", type=int, default=120)
    parser.add_argument("--n-non-baby", type=int, default=120)
    parser.add_argument("--size", type=int, nargs=2, default=(160, 160), metavar=("H", "W"))
    args = parser.parse_args()

    baby_dir = args.data_root / "baby"
    nb_dir = args.data_root / "non_baby"
    baby_dir.mkdir(parents=True, exist_ok=True)
    nb_dir.mkdir(parents=True, exist_ok=True)

    h, w = int(args.size[0]), int(args.size[1])
    for i in range(args.n_baby):
        write_random_face_crop(baby_dir / f"dummy_baby_{i:05d}.jpg", seed=i, size=(h, w))
    for i in range(args.n_non_baby):
        write_random_face_crop(nb_dir / f"dummy_nb_{i:05d}.jpg", seed=10_000 + i, size=(h, w))

    print(f"Wrote {args.n_baby} JPGs -> {baby_dir}")
    print(f"Wrote {args.n_non_baby} JPGs -> {nb_dir}")
    print("Next: python train.py --data-root data --device cpu")


if __name__ == "__main__":
    main()
