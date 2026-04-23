import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

class CNN_I5(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一層：處理 32x15
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5,3),
                       stride=(1,1), 
                       padding=(2,1), 
                       dilation=(1,1)),
            nn.BatchNorm2d(64), # 建議先用 BatchNorm，容錯率較高
            nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 1))
        )
        # 第二層
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3),
                       stride=(1,1), 
                       padding=(2,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 1))
        )
        
        # 全連接層：這裡的 16384 需要根據最後輸出的尺寸計算
        # 32 -> MaxPool -> 16 -> MaxPool -> 8
        # 15 -> 不變 -> 15 -> 不變 -> 15
        # 128 * 8 * 15 = 15360
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 15, 2)
        )

    def forward(self, x):
        # x 的輸入應該是 [Batch, 1, 32, 15]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x


class CNN_I20(nn.Module):
    def __init__(self):
        super(CNN_I20, self).__init__()
        
        # 第一層：處理 64x60 (高x寬)
        # 論文規格通常使用 Stride 來進行下採樣，而非純靠 Pooling
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), 
                      stride=(3, 1), 
                      padding=(4, 1), 
                      dilation=(2, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        
        # 第二層
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3),
                       stride=(1, 1), 
                       padding=(2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # 第三層 (增加深度以處理 20 天的複雜形態)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), 
                      stride=(1, 1), 
                      padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01)
        )

        # 核心計算：全連接層維度
        # 輸入 64x60 -> Layer1: 11x60 -> Layer2: 5x60 -> Layer3: 5x60
        # 最終維度: 256 * 5 * 60 = 76800
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(76800, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # 確保輸入維度為 [Batch, 1, 64, 60]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x


def set_training_seed(seed: int):
    """Set random seeds used by numpy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(I: int):
    """Create the CNN model that matches the image window length."""
    if I == 5:
        return CNN_I5()
    if I == 20:
        return CNN_I20()
    raise ValueError("Only I=5 and I=20 are currently supported by model.py")


def _fit_pixel_norm(X_train: np.ndarray):
    X_float = X_train.astype(np.float32, copy=False)
    mean = float(np.mean(X_float))
    std = float(np.std(X_float))
    if std < 1e-7:
        std = 1.0
    return mean, std


def _apply_pixel_norm(X: np.ndarray, mean: float, std: float):
    X_float = X.astype(np.float32, copy=False)
    return ((X_float - mean) / (std + 1e-7)).astype(np.float32, copy=False)


def _make_loader(X, y, batch_size=128, shuffle=False, num_workers=0):
    ds = TensorDataset(
        torch.from_numpy(X[:, np.newaxis, :, :]),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def make_resplit_dataloaders(
    X_trainval,
    y_trainval,
    val_ratio=0.3,
    seed=42,
    batch_size=128,
    num_workers=0,
):
    """Split trainval with a seed and return train/val DataLoaders."""
    indices = np.arange(len(y_trainval))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=y_trainval,
    )

    mean, std = _fit_pixel_norm(X_trainval[train_idx])
    X_train_n = _apply_pixel_norm(X_trainval[train_idx], mean, std)
    X_val_n = _apply_pixel_norm(X_trainval[val_idx], mean, std)

    train_loader = _make_loader(
        X_train_n,
        y_trainval[train_idx],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = _make_loader(
        X_val_n,
        y_trainval[val_idx],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    split_info = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "pixel_mean": mean,
        "pixel_std": std,
    }
    return train_loader, val_loader, split_info


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))

    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_resplit_model(
    X_trainval,
    y_trainval,
    I,
    R,
    seed,
    save_dir="models",
    Market="US",
    val_ratio=0.3,
    epochs=50,
    batch_size=128,
    lr=1e-4,
    weight_decay=1e-4,
    num_workers=0,
    device=None,
    monitor="val_acc",
):
    """
    Train one model with a seed-specific train/val split and initialization.
    The best model checkpoint for this seed is saved to disk.
    """
    set_training_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, split_info = make_resplit_dataloaders(
        X_trainval,
        y_trainval,
        val_ratio=val_ratio,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = make_model(I).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    best_score = -float("inf") if monitor == "val_acc" else float("inf")
    best_path = os.path.join(save_dir, f"{Market}_best_model_I{I}R{R}_seed{seed}.pth")
    history = []

    epoch_bar = tqdm(
        range(epochs),
        desc=f"seed {seed}",
        leave=False,
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / max(len(train_loader), 1)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append({
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        score = val_acc if monitor == "val_acc" else val_loss
        improved = score > best_score if monitor == "val_acc" else score < best_score
        if improved:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "I": I,
                    "R": R,
                    "seed": seed,
                    "pixel_mean": split_info["pixel_mean"],
                    "pixel_std": split_info["pixel_std"],
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                    "train_idx": split_info["train_idx"],
                    "val_idx": split_info["val_idx"],
                },
                best_path,
            )

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc * 100:.2f}%",
            best=f"{best_score * 100:.2f}%" if monitor == "val_acc" else f"{best_score:.4f}",
        )

    return {
        "seed": seed,
        "best_path": best_path,
        "best_score": best_score,
        "history": history,
    }


def train_resplit_ensemble(
    X_trainval,
    y_trainval,
    I,
    R,
    seeds=(0, 1, 2, 3, 4),
    save_dir="models",
    Market="US",
    val_ratio=0.3,
    epochs=50,
    batch_size=128,
    lr=1e-4,
    weight_decay=1e-4,
    num_workers=0,
    device=None,
):
    """
    Train multiple models. Each seed uses a different train/val split and
    a different initialization. Each seed's best checkpoint is saved.
    """
    results = []
    seed_bar = tqdm(
        list(seeds),
        desc="ensemble runs",
        dynamic_ncols=True,
    )
    for seed in seed_bar:
        seed_bar.set_postfix(seed=seed)
        result = train_one_resplit_model(
            X_trainval=X_trainval,
            y_trainval=y_trainval,
            I=I,
            R=R,
            seed=seed,
            save_dir=save_dir,
            Market=Market,
            val_ratio=val_ratio,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            num_workers=num_workers,
            device=device,
        )
        results.append(result)
    return results


def predict_proba_from_checkpoint(X, checkpoint_path, batch_size=256, num_workers=0, device=None):
    """Predict P(label=1) using a saved ensemble checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = make_model(int(checkpoint["I"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_n = _apply_pixel_norm(X, checkpoint["pixel_mean"], checkpoint["pixel_std"])
    loader = _make_loader(
        X_n,
        np.zeros(len(X_n), dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            p = torch.softmax(outputs, dim=1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def average_ensemble_predictions(X, checkpoint_paths, batch_size=256, num_workers=0, device=None):
    """Average P(label=1) across multiple saved model checkpoints."""
    preds = [
        predict_proba_from_checkpoint(
            X,
            path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        for path in checkpoint_paths
    ]
    return np.mean(np.vstack(preds), axis=0)
