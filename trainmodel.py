# train.py

import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torch.cuda.amp import autocast, GradScaler

from dataset import ChickenVoiceDataset
from model import get_model


# ----------------- simple mel augment wrapper -----------------
class AugmentedDataset(Dataset):
    """
    Wraps a Dataset/Subset and applies light augmentation on mel spectrograms.
    Input sample: (mel_tensor [1, n_mels, T], label)
    """

    def __init__(self, base_dataset, time_mask_prob=0.3, freq_mask_prob=0.3, noise_prob=0.3):
        self.base_dataset = base_dataset
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        mel, label = self.base_dataset[idx]  # mel: [1, n_mels, T]
        mel = mel.clone()

        _, n_mels, T = mel.shape

        # ---- time mask ----
        if random.random() < self.time_mask_prob and T > 10:
            t = random.randint(0, T - 1)
            width = random.randint(1, max(1, T // 10))
            t_end = min(T, t + width)
            mel[:, :, t:t_end] = 0.0

        # ---- freq mask ----
        if random.random() < self.freq_mask_prob and n_mels > 10:
            f = random.randint(0, n_mels - 1)
            height = random.randint(1, max(1, n_mels // 8))
            f_end = min(n_mels, f + height)
            mel[:, f:f_end, :] = 0.0

        # ---- add small Gaussian noise ----
        if random.random() < self.noise_prob:
            noise = torch.randn_like(mel) * 0.05
            mel = mel + noise

        return mel, label


# ----------------- dataloaders -----------------
def get_dataloaders(
    data_dir="data",
    batch_size=16,
    val_ratio=0.1,      # 10% val, 90% train
    num_workers=0,      # 0 is safest on Windows/CPU
    use_augmentation=True,
):
    base_dataset = ChickenVoiceDataset(root_dir=data_dir)
    n_total = len(base_dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        base_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    if use_augmentation:
        train_ds = AugmentedDataset(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,   # True not needed on pure CPU
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, base_dataset.label2idx


# ----------------- train / eval loops -----------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda_amp = device.startswith("cuda")

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        if use_cuda_amp:
            with autocast():
                outputs = model(xb)
                loss = criterion(outputs, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda_amp = device.startswith("cuda")

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if use_cuda_amp:
                with autocast():
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
            else:
                outputs = model(xb)
                loss = criterion(outputs, yb)

            running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ----------------- main train script -----------------
def main():
    # --------- hyperparams (Option B: stronger) ---------
    data_dir = "data"
    batch_size = 16
    num_epochs = 20
    lr = 1e-3
    val_ratio = 0.1           # 90% train, 10% val
    num_workers = 0           # 0 = safest on Windows/CPU
    use_augmentation = True
    # ---------------------------------------------------

    train_loader, val_loader, label2idx = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=num_workers,
        use_augmentation=use_augmentation,
    )
    num_classes = len(label2idx)
    print("Label mapping:", label2idx)

    model, device = get_model(num_classes=num_classes)
    print("Training on device:", device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # LR scheduler: reduce LR if val_acc plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    scaler = GradScaler(enabled=device.startswith("cuda"))

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "chicken_resnet_best.pt")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        print(f"  Train | loss: {train_loss:.4f}  acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        print(f"  Val   | loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "label2idx": label2idx,
                },
                best_model_path,
            )
            print(
                f"  ✅ New best model saved to {best_model_path} "
                f"(val_acc={val_acc:.4f})"
            )

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)


if __name__ == "__main__":
    main()
