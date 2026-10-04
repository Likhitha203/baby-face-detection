"""Confusion matrix, per-class metrics, ROC, optional Platt-style calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from augment import get_val_transforms
from dataset import BabyFaceDataset
from modeling import build_baby_classifier, load_classifier_ckpt


def collect_logits_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y_true (N,), prob_baby (N,), logits (N,2). Label 0 = baby."""
    model.eval()
    ys: list[np.ndarray] = []
    pb: list[np.ndarray] = []
    lg: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob_baby = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
            ys.append(y.numpy())
            pb.append(prob_baby)
            lg.append(logits.cpu().numpy())
    return np.concatenate(ys), np.concatenate(pb), np.concatenate(lg)


def platt_on_margin(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    logits_tgt: np.ndarray,
) -> np.ndarray:
    """Calibrate baby probability via logistic regression on logits margin."""
    margins = logits_val[:, 0] - logits_val[:, 1]
    margins_tgt = logits_tgt[:, 0] - logits_tgt[:, 1]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(margins.reshape(-1, 1), np.where(y_val == 0, 1, 0))
    return lr.predict_proba(margins_tgt.reshape(-1, 1))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/baby_classifier.pth"))
    parser.add_argument(
        "--indices",
        type=Path,
        default=Path("models/test_split_indices.pt"),
        help="Index list saved by train.py (test split).",
    )
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--platt-from-val", action="store_true")
    parser.add_argument(
        "--val-indices",
        type=Path,
        default=Path("models/val_split_indices.pt"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval_out"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    ds = BabyFaceDataset(args.data_root, transform=get_val_transforms())

    try:
        indices = torch.load(args.indices, map_location="cpu", weights_only=False)
    except TypeError:
        indices = torch.load(args.indices, map_location="cpu")
    if isinstance(indices, torch.Tensor):
        indices_list = indices.tolist()
    else:
        indices_list = list(indices)
    subset = torch.utils.data.Subset(ds, indices_list)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_baby_classifier(pretrained=False)
    load_classifier_ckpt(model, args.checkpoint, map_location=device)
    model.to(device)

    y_true, prob_baby, logits_test = collect_logits_probs(model, loader, device)

    prob_pos = prob_baby
    if args.platt_from_val and args.val_indices.is_file():
        try:
            val_idx = torch.load(args.val_indices, map_location="cpu", weights_only=False)
        except TypeError:
            val_idx = torch.load(args.val_indices, map_location="cpu")
        val_list = val_idx.tolist() if isinstance(val_idx, torch.Tensor) else list(val_idx)
        val_subset = torch.utils.data.Subset(ds, val_list)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        y_val, _, logits_val = collect_logits_probs(model, val_loader, device)
        prob_pos = platt_on_margin(logits_val, y_val, logits_test)

    y_pred = np.where(prob_pos >= args.threshold, 0, 1)
    target_names = ["baby", "non_baby"]
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=target_names, digits=4))

    fp = np.logical_and(y_true != 0, y_pred == 0).sum()
    fn = np.logical_and(y_true == 0, y_pred != 0).sum()
    tp = np.logical_and(y_true == 0, y_pred == 0).sum()
    tn = np.logical_and(y_true != 0, y_pred != 0).sum()
    baby_recall = tp / max(tp + fn, 1)
    baby_prec = tp / max(tp + fp, 1)
    print(f"Baby recall={baby_recall:.4f}  Baby precision={baby_prec:.4f}  TP={tp} FN={fn} FP={fp} TN={tn}")

    y_binary = np.where(y_true == 0, 1, 0)
    fpr, tpr, _ = roc_curve(y_binary, prob_pos)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC (baby positive)={roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — baby vs rest")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = args.out_dir / "roc.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"Saved {roc_path.resolve()}")


if __name__ == "__main__":
    main()
