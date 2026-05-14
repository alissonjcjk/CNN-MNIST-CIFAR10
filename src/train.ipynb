"""
train.py
========
Pipeline de treino/validação genérico.
Suporta MNIST e CIFAR-10 com a CustomCNN.

Uso:
    python src/train.py --dataset mnist --epochs 20
    python src/train.py --dataset cifar10 --epochs 30
"""

import argparse
import platform
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CustomCNN

# ──────────────────────────────────────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
METRIC_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")

os.makedirs(DATA_DIR, exist_ok=True)
NUM_WORKERS = 0 if platform.system() == "Windows" else 2
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ──────────────────────────────────────────────────────────────────────────────
def get_dataloaders(dataset_name: str, batch_size: int = 128):
    """Retorna train_loader, val_loader, test_loader e metadados do dataset."""
    if dataset_name == "mnist":
        mean, std = (0.1307,), (0.3081,)
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_transform)
        test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
        in_channels, input_size = 1, 28

    elif dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)
        in_channels, input_size = 3, 32
    else:
        raise ValueError(f"Dataset '{dataset_name}' não suportado. Use 'mnist' ou 'cifar10'.")

    # Split treino / validação (85% / 15%)
    val_size = int(0.15 * len(train_full))
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, in_channels, input_size


# ──────────────────────────────────────────────────────────────────────────────
# Funções de treino / avaliação
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_curves(history: dict, dataset_name: str):
    """Gráfico de loss e accuracy por época."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title(f"Loss — {dataset_name.upper()}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title(f"Accuracy — {dataset_name.upper()}")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"custom_cnn_{dataset_name}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Curvas salvas em {path}")


def plot_confusion_matrix(labels, preds, class_names, dataset_name):
    import seaborn as sns
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {dataset_name.upper()}")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"custom_cnn_{dataset_name}_confusion.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Matriz de confusão salva em {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Treina CustomCNN em MNIST ou CIFAR-10")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["mnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset.upper()}")
    print(f"  Device  : {DEVICE}")
    print(f"  Épocas  : {args.epochs}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader, in_channels, input_size = \
        get_dataloaders(args.dataset, args.batch_size)

    # Classes
    if args.dataset == "mnist":
        class_names = [str(i) for i in range(10)]
    else:
        class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    # Modelo
    model = CustomCNN(
        in_channels=in_channels,
        num_classes=10,
        input_size=input_size,
        dropout_rate=args.dropout,
    ).to(DEVICE)
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Histórico
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"Época {epoch:03d}/{args.epochs:03d}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CKPT_DIR, f"custom_cnn_{args.dataset}_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [★] Novo melhor modelo salvo! Val Acc = {best_val_acc:.4f}")

    elapsed = time.time() - start
    print(f"\nTreino concluído em {elapsed/60:.1f} min")

    # Avaliação final no teste
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n[TEST] Accuracy = {test_acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Salva métricas
    metrics = {
        "dataset": args.dataset,
        "test_accuracy": round(test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs": args.epochs,
        "training_time_min": round(elapsed / 60, 2),
        "history": history,
    }
    metric_path = os.path.join(METRIC_DIR, f"custom_cnn_{args.dataset}.json")
    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[✓] Métricas salvas em {metric_path}")

    # Plots
    plot_curves(history, args.dataset)
    plot_confusion_matrix(test_labels, test_preds, class_names, args.dataset)


if __name__ == "__main__":
    main()
