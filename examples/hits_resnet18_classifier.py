"""Training script for binary classification of T-Q images stored in HDF5.

The script expects an HDF5 file with datasets:
    images: shape [N_events, C, Q_bins, T_bins] (float32)
    labels: shape [N_events] with integer class labels 0 or 1

The model is a modified torchvision ResNet18 that adapts the input
channels and final classification layer for binary classification.
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models


class HitsH5Dataset(Dataset):
    """Dataset for lazily reading T-Q images and labels from an HDF5 file.

    Args:
        h5_path: Path to the HDF5 file containing ``images`` and ``labels`` datasets.

    The dataset supports multiprocessing data loading by lazily opening
    the file handle in each worker process when ``__getitem__`` is first
    invoked.
    """

    def __init__(self, h5_path: str) -> None:
        self.h5_path = h5_path
        self._file: h5py.File | None = None
        self._images = None
        self._labels = None
        self._ensure_open()
        # Cache the number of channels for downstream model initialization
        self.num_channels = int(self._images.shape[1])

    def _ensure_open(self) -> None:
        """Ensure the HDF5 file is open in the current process."""
        if self._file is None or not isinstance(self._file, h5py.File):
            self._file = h5py.File(self.h5_path, "r")
            self._images = self._file["images"]
            self._labels = self._file["labels"]

    def __len__(self) -> int:  # pragma: no cover - trivial
        self._ensure_open()
        return int(self._images.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        image = torch.tensor(self._images[idx], dtype=torch.float32)
        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return image, label

    def __del__(self) -> None:  # pragma: no cover - cleanup
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


class ResNet18Classifier(nn.Module):
    """ResNet18-based classifier for binary prediction.

    Args:
        in_channels: Number of channels in the input T-Q images.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Avoid downloading pretrained weights by setting weights=None
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, C, Q_bins, T_bins].

        Returns:
            Logits tensor of shape [B, 2].
        """
        return self.model(x)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary classification of T-Q images with ResNet18")
    parser.add_argument("--data_h5_path", type=str, required=False, default="data/dataset.h5", help="Path to the HDF5 dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of data used for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, default="best_model.pth", help="Path to save the best model state_dict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.data_h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {args.data_h5_path}")

    dataset = HitsH5Dataset(args.data_h5_path)
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier(dataset.num_channels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_path)
            print(f"New best model saved to {args.output_path} with val acc {best_val_acc:.4f}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
