#!/usr/bin/env python3
"""
Download pretrained assets:

- Ultralytics publishes face models in community repos; mirror your chosen
  `yolov8n-face.pt` manually if this URL expires.
- Train `baby_classifier.pth` with train.py once `data/` is populated.

Uses urllib; no optional deps.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen


DEFAULT_FACE_URL = (
    "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt"
)


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo-url",
        default=DEFAULT_FACE_URL,
        help="Direct URL for yolov8n-face.pt (Ultralytics compatible).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/yolov8n-face.pt"),
        help="Destination for face detector weights.",
    )
    args = parser.parse_args()

    fetch(args.yolo_url, args.out)
    print(f"Saved {args.out.resolve()}. Train classifier with: python train.py")


if __name__ == "__main__":
    main()
