"""Fine-tune MobileNetV3-Large for baby vs non-baby face crops."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from augment import get_train_transforms, get_val_transforms
from dataset import BabyFaceDataset
from modeling import build_baby_classifier


LABEL_BABY = 0


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def stratified_counts(dataset: BabyFaceDataset) -> tuple[int, int]:
    n_baby = sum(1 for _, y in dataset.samples if y == LABEL_BABY)
    n_non = len(dataset) - n_baby
    return n_baby, n_non


def stratified_split_indices(
    dataset: BabyFaceDataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    baby_idx = [i for i, (_, y) in enumerate(dataset.samples) if y == LABEL_BABY]
    non_idx = [i for i, (_, y) in enumerate(dataset.samples) if y != LABEL_BABY]

    def split_group(idxs: list[int]) -> tuple[list[int], list[int], list[int]]:
        a = idxs.copy()
        rng.shuffle(a)
        n = len(a)
        n_train = math.floor(n * train_frac)
        n_val = math.floor(n * val_frac)
        n_test = max(0, n - n_train - n_val)
        tr = a[:n_train]
        va = a[n_train : n_train + n_val]
        te = a[n_train + n_val :] if n_test else a[n_train + n_val :]
        return tr, va, te

    bt, bv, bs = split_group(baby_idx)
    nt, nv, ns = split_group(non_idx)

    train_i = bt + nt
    val_i = bv + nv
    test_i = bs + ns
    rng.shuffle(train_i)
    rng.shuffle(val_i)
    rng.shuffle(test_i)
    return train_i, val_i, test_i


def make_weighted_sampler(
    train_indices: list[int],
    dataset: BabyFaceDataset,
    baby_weight_mult: float,
) -> WeightedRandomSampler:
    weights: list[float] = []
    for idx in train_indices:
        _, y = dataset.samples[idx]
        weights.append(baby_weight_mult if y == LABEL_BABY else 1.0)
    tw = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(tw, num_samples=len(tw), replacement=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    acc = correct / max(n, 1)
    return total_loss / max(n, 1), acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--baby-sampler-mult", type=float, default=1.5)
    parser.add_argument("--freeze-blocks", type=int, default=12)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=Path("models/baby_classifier.pth"))
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = args.data_root.resolve()
    ds_index = BabyFaceDataset(data_root, transform=None)
    if len(ds_index) == 0:
        raise SystemExit(
            f"No images under {data_root}/baby or {data_root}/non_baby.\n"
            "Add face crops (.jpg/.png/…) there, or for a smoke test run:\n"
            "  python bootstrap_dummy_data.py --data-root data"
        )

    n_baby, n_non = stratified_counts(ds_index)
    print(f"Images: baby={n_baby}, non_baby={n_non}")

    train_i, val_i, test_i = stratified_split_indices(ds_index, seed=args.seed)
    print(f"Split sizes: train={len(train_i)} val={len(val_i)} test={len(test_i)}")

    train_view = BabyFaceDataset(data_root, transform=get_train_transforms())
    val_view = BabyFaceDataset(data_root, transform=get_val_transforms())

    pin_mem = args.device == "cuda" and torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_view, train_i),
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_i, ds_index, args.baby_sampler_mult),
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        Subset(val_view, val_i),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA unavailable; using CPU")
    elif (
        args.device == "mps"
        and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available())
    ):
        device = torch.device("cpu")
        print("MPS unavailable; using CPU")
    else:
        device = torch.device(args.device)

    model = build_baby_classifier(num_classes=2, freeze_blocks=args.freeze_blocks).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("-inf")
    patience_left = args.early_stop_patience

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  LR={scheduler.get_last_lr()[0]:.2e}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            patience_left = args.early_stop_patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "label_baby_idx": LABEL_BABY,
                    "architecture": "mobilenet_v3_large_two_head",
                },
                args.save_path,
            )
            print(f"  Saved checkpoint to {args.save_path}")
        else:
            patience_left -= 1
            print(f"  No improvement  early_stop_remaining={patience_left}")
            if patience_left <= 0:
                print("Early stopping.")
                break

    print(f"\nBest val_acc={best_val:.4f}  Checkpoint: {args.save_path}")

    ta = torch.tensor(train_i, dtype=torch.long)
    va = torch.tensor(val_i, dtype=torch.long)
    te = torch.tensor(test_i, dtype=torch.long)
    torch.save(ta, args.save_path.parent / "train_split_indices.pt")
    torch.save(va, args.save_path.parent / "val_split_indices.pt")
    torch.save(te, args.save_path.parent / "test_split_indices.pt")


if __name__ == "__main__":
    main()
