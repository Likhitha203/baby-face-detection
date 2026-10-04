"""Real-time webcam: YOLOv8-face detects faces, MobileNetV3 scores baby/non-baby."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:
    raise SystemExit(
        "No OpenCV bindings (cv2). Use the SAME interpreter you use to run this script:\n"
        "  python -m pip uninstall opencv-python opencv-python-headless -y\n"
        "  python -m pip install --no-cache-dir opencv-python\n"
        "Avoid plain `pip install` if multiple Pythons are on PATH."
    ) from None

import numpy as np

# Ultralytics patches cv2.imshow; keep real highgui entry points for display + probe.
_cv2_namedWindow = cv2.namedWindow
_cv2_imshow = cv2.imshow
_cv2_waitKey = cv2.waitKey
_cv2_destroyWindow = cv2.destroyWindow
_cv2_window_autosize = cv2.WINDOW_AUTOSIZE

import torch
from ultralytics import YOLO

from augment import get_val_transforms
from modeling import build_baby_classifier, load_classifier_ckpt

try:
    import dlib

    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False


def opencv_gui_works() -> bool:
    """False for opencv-python-headless / builds without Qt or Win32 highgui."""
    probe = "__baby_det_gui_probe__"
    try:
        dummy = np.zeros((4, 4, 3), dtype=np.uint8)
        _cv2_namedWindow(probe, _cv2_window_autosize)
        _cv2_imshow(probe, dummy)
        _cv2_waitKey(1)
        _cv2_destroyWindow(probe)
        return True
    except cv2.error:
        try:
            _cv2_destroyWindow(probe)
        except cv2.error:
            pass
        return False


def open_video_writer(
    path: Path,
    frame_size: tuple[int, int],
    fps: float,
) -> tuple[cv2.VideoWriter | None, str]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frame_size
    fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc_mp4, fps, (w, h))
    if not writer.isOpened():
        avi = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        if writer.isOpened():
            return writer, str(avi)
        return None, ""
    return writer, str(path)


def padded_crop_xyxy(frame: np.ndarray, xyxy, padding_ratio: float) -> np.ndarray | None:
    x1, y1, x2, y2 = map(float, xyxy)
    h, w = frame.shape[:2]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    nx1 = int(max(0, x1 - pad_x))
    ny1 = int(max(0, y1 - pad_y))
    nx2 = int(min(w - 1, x2 + pad_x))
    ny2 = int(min(h - 1, y2 + pad_y))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return frame[ny1 : ny2, nx1:nx2]


def apply_horizontal_shifts(rgb: np.ndarray, shifts: tuple[int, ...]) -> list[np.ndarray]:
    h, w = rgb.shape[:2]
    out: list[np.ndarray] = [rgb.copy()]
    for s in shifts:
        if s == 0:
            continue
        m = np.float32([[1, 0, float(s)], [0, 1, 0]])
        shifted = cv2.warpAffine(
            rgb, m, (w, h), borderMode=cv2.BORDER_REFLECT_101,
        )
        out.append(shifted)
    return out


class BabyClassifierTorch:
    def __init__(self, ckpt_path: Path, device: torch.device):
        self.device = device
        self.model = build_baby_classifier(pretrained=False).to(device)
        load_classifier_ckpt(self.model, ckpt_path, map_location=device)
        self.model.eval()
        self.aug = get_val_transforms()

    @torch.no_grad()
    def prob_baby(self, crop_bgr: np.ndarray, tta_px: int = 0) -> tuple[float, float]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crops = (
            apply_horizontal_shifts(rgb, (0,))
            if tta_px <= 0
            else apply_horizontal_shifts(rgb, (0, -tta_px, tta_px))
        )
        logits_stack: list[torch.Tensor] = []
        for crop in crops:
            t = self.aug(image=crop)["image"].unsqueeze(0).to(self.device)
            logits_stack.append(self.model(t))
        logits = torch.stack(logits_stack, dim=0).mean(dim=0)
        prob = torch.softmax(logits, dim=1)[0, 0].item()
        return prob, 1.0 - prob


def maybe_align_face_dlib(crop_bgr: np.ndarray, predictor: Any) -> np.ndarray:
    if predictor is None or not HAS_DLIB:
        return crop_bgr
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rect = dlib.rectangle(0, 0, w - 1, h - 1)
    shape = predictor(gray, rect)
    if shape.num_parts < 46:
        return crop_bgr
    le = np.array([shape.part(36).x, shape.part(36).y], dtype=np.float32)
    re = np.array([shape.part(45).x, shape.part(45).y], dtype=np.float32)
    center = ((le[0] + re[0]) * 0.5, (le[1] + re[1]) * 0.5)
    dx = float(re[0] - le[0])
    dy = float(re[1] - le[1])
    angle = -math.degrees(math.atan2(dy, dx))
    m_rot = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(
        crop_bgr, m_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )


def draw_simple_face_result(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    is_baby: bool,
) -> None:
    """Thin gray box + 'Not baby', or bold green box + 'Baby' — no percentages."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    pad_x, pad_y = 8, 5

    if is_baby:
        border = (0, 255, 0)
        border_thick = 3
        text = "Baby"
        bg = (0, 180, 0)
        fg = (255, 255, 255)
    else:
        border = (120, 120, 120)
        border_thick = 1
        text = "Not baby"
        bg = (45, 45, 48)
        fg = (235, 235, 235)

    cv2.rectangle(frame, (x1, y1), (x2, y2), border, border_thick)

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y_text = max(2, y1 - th - 2 * pad_y - 2)
    x_left = max(2, x1)
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y
    if x_left + box_w > frame.shape[1] - 2:
        x_left = max(2, frame.shape[1] - box_w - 2)
    cv2.rectangle(
        frame,
        (x_left, y_text),
        (x_left + box_w, y_text + box_h),
        bg,
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x_left + pad_x, y_text + th + pad_y - 3),
        font,
        scale,
        fg,
        thickness,
        cv2.LINE_AA,
    )


def draw_global_hud(
    frame: np.ndarray,
    lines: list[str],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    pad = 10
    sizes = [cv2.getTextSize(t, font, scale, thick) for t in lines]
    max_w = max(s[0][0] for s in sizes)
    line_h = max(s[0][1] for s in sizes)
    box_w = min(max_w + 2 * pad, frame.shape[1] - 2)
    box_h = len(lines) * (line_h + 6) + pad
    cv2.rectangle(frame, (2, 2), (box_w + 4, box_h + 8), (22, 24, 32), -1)
    y = pad + line_h + 2
    for text in lines:
        cv2.putText(frame, text, (pad + 2, y), font, scale, (235, 235, 240), thick, cv2.LINE_AA)
        y += line_h + 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", type=Path, default=Path("models/yolov8n-face.pt"))
    parser.add_argument("--classifier", type=Path, default=Path("models/baby_classifier.pth"))
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO face confidence threshold.")
    parser.add_argument("--baby-threshold", type=float, default=0.82)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tta-px", type=int, default=0, help="If >0, average 3 horizontal shifts (±px).")
    parser.add_argument(
        "--dlib-predictor",
        type=Path,
        default=None,
        help="Path to shape_predictor_68_face_landmarks.dat (optional).",
    )
    parser.add_argument("--bench", action="store_true", help="Print per-frame timing to stdout.")
    parser.add_argument(
        "--debug-hud",
        action="store_true",
        help="Show FPS, face count, and threshold in corner (normally hidden).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Never open a preview window (headless); requires --output to save video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save annotated MP4/AVI while running (combine with live preview by default).",
    )
    args = parser.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
        print("CUDA unavailable; inference on CPU")
    elif dev == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        dev = "cpu"
        print("MPS unavailable; inference on CPU")
    device = torch.device(dev)

    if not args.yolo.is_file():
        raise SystemExit(
            f"Missing {args.yolo}. Download a YOLOv8n-face weight (e.g. community yolov8n-face.pt) into models/."
        )
    if not args.classifier.is_file():
        raise SystemExit(f"Missing {args.classifier}. Train with train.py or place checkpoint in models/.")

    yolo = YOLO(str(args.yolo))
    clf = BabyClassifierTorch(args.classifier, device)

    predictor = None
    if args.dlib_predictor is not None:
        if not HAS_DLIB:
            print("dlib not installed; alignment disabled.")
        elif not args.dlib_predictor.is_file():
            print(f"Predictor not found at {args.dlib_predictor}; alignment disabled.")
        else:
            predictor = dlib.shape_predictor(str(args.dlib_predictor))

    if args.no_show and args.output is None:
        raise SystemExit("--no-show requires --output path (e.g. --output runs/out.mp4).")

    if not args.no_show and not opencv_gui_works():
        raise SystemExit(
            "OpenCV cannot display a live window (common with opencv-python-headless).\n"
            "Use the SAME Python you run inference with:\n"
            "  python -m pip uninstall opencv-python-headless -y\n"
            "  python -m pip install --upgrade opencv-python\n"
            "If you then get 'No module named cv2', reinstall the wheel files:\n"
            "  python -m pip uninstall opencv-python -y\n"
            "  python -m pip install --no-cache-dir opencv-python\n\n"
            "Headless recording only:\n"
            "  python inference.py --no-show --output runs/out.mp4\n"
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps < 5 or fps > 120:
        fps = 30.0

    use_gui = not args.no_show

    writer: cv2.VideoWriter | None = None
    writer_path = ""
    if args.output is not None:
        writer, writer_path = open_video_writer(args.output, (fw, fh), fps)
        if writer is None:
            raise SystemExit(
                "Could not open VideoWriter. Try another --output path or use .avi (XVID)."
            )
        if use_gui:
            print(f"Recording to {writer_path} (with live preview).")
        else:
            print(f"Recording to {writer_path} — stop with Ctrl+C.")

    win = "Baby Detector"
    consecutive_bad = 0
    had_frame = False
    ema_fps = -1.0
    try:
        if use_gui:
            _cv2_namedWindow(win, _cv2_window_autosize)
            print("Live camera — close window or press Q / ESC to quit.")

        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                consecutive_bad += 1
                if had_frame and consecutive_bad > 120:
                    print("Camera stream ended or device disconnected.")
                    break
                if not had_frame and consecutive_bad > 400:
                    print("No frames from camera; check index and permissions.")
                    break
                time.sleep(0.02)
                continue
            consecutive_bad = 0
            had_frame = True

            results = yolo.predict(
                source=frame,
                conf=args.conf,
                verbose=False,
                device=(
                    device.type if device.type in ("cuda", "mps") else "cpu"
                ),
            )
            t_yolo = time.perf_counter()

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                for box in xyxy:
                    crop = padded_crop_xyxy(frame, box, args.padding)
                    if crop is None or crop.size == 0:
                        continue
                    if predictor is not None:
                        crop = maybe_align_face_dlib(crop, predictor)
                    pb, _pn = clf.prob_baby(crop, tta_px=args.tta_px)
                    x1, y1, x2, y2 = map(int, box)

                    is_baby = pb >= args.baby_threshold
                    draw_simple_face_result(frame, x1, y1, x2, y2, is_baby=is_baby)

            t_end = time.perf_counter()

            if args.debug_hud:
                n_faces_det = 0
                for r in results:
                    if r.boxes is not None:
                        n_faces_det += len(r.boxes)
                elapsed = max(t_end - t0, 1e-9)
                inst_fps = 1.0 / elapsed
                ema_fps = inst_fps if ema_fps < 0 else ema_fps * 0.88 + inst_fps * 0.12
                draw_global_hud(
                    frame,
                    [
                        f"FPS ~{ema_fps:.1f}",
                        f"Faces: {n_faces_det}",
                        "(debug) Q / ESC quit",
                    ],
                )

            if args.bench:
                nf = sum(len(r.boxes) for r in results if r.boxes is not None)
                print(
                    f"yolo_ms={(t_yolo - t0) * 1000:.1f}  "
                    f"total_ms={(t_end - t0) * 1000:.1f}  n_faces={nf}",
                )
                writer.write(frame)
            if use_gui:
                _cv2_imshow(win, frame)
                key = _cv2_waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if use_gui:
            try:
                _cv2_destroyWindow(win)
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
