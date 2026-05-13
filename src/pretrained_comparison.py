"""
pretrained_comparison.py
========================
Avalia uma rede pré-treinada (ResNet18 fine-tuned) em MNIST e CIFAR-10
e salva as métricas para comparação com a CustomCNN.

Uso:
    python src/pretrained_comparison.py --dataset cifar10
    python src/pretrained_comparison.py --dataset mnist
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
METRIC_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

EPOCHS_FINETUNE = 10  # Poucas épocas: pesos pré-treinados já são bons


# ──────────────────────────────────────────────────────────────────────────────
def build_resnet18(num_classes: int = 10, in_channels: int = 3):
    """
    ResNet18 pré-treinada no ImageNet com cabeça FC substituída.
    Para MNIST (1 canal), duplicamos o canal de entrada para 3.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Substitui o classificador final
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Congela todas as camadas exceto a última (fine-tuning eficiente)
    for name, param in model.named_parameters():
        if "fc" not in name and "layer4" not in name:
            param.requires_grad = False
    return model


def get_dataloaders(dataset_name: str, batch_size: int = 64):
    """
    Para ResNet, as imagens precisam ter pelo menos 32x32 e 3 canais.
    MNIST: upscale para 32x32, repetição para 3 canais via Grayscale→RGB.
    """
    if dataset_name == "mnist":
        mean = (0.485, 0.456, 0.406)   # ImageNet stats
        std = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_transform)
        test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)

    elif dataset_name == "cifar10":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Dataset '{dataset_name}' não suportado.")

    val_size = int(0.15 * len(train_full))
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = out.max(1)
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
        out = model(images)
        loss = criterion(out, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


def plot_curves(history, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title(f"ResNet18 Loss — {dataset_name.upper()}")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title(f"ResNet18 Accuracy — {dataset_name.upper()}")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"resnet18_{dataset_name}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Curvas salvas em {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["mnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=EPOCHS_FINETUNE)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ResNet18 (ImageNet) → Fine-tune em {args.dataset.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, args.batch_size)

    if args.dataset == "mnist":
        class_names = [str(i) for i in range(10)]
    else:
        class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    model = build_resnet18(num_classes=10).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params : {total_params:,}")
    print(f"  Treináveis   : {trainable_params:,} ({100*trainable_params/total_params:.1f}%)\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_path = os.path.join(CKPT_DIR, f"resnet18_{args.dataset}_best.pth")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"Época {epoch:02d}/{args.epochs}")
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl, va, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"  Train Loss: {tl:.4f}  Acc: {ta:.4f}")
        print(f"  Val   Loss: {vl:.4f}  Acc: {va:.4f}")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [★] Melhor modelo salvo! Val Acc = {best_val_acc:.4f}")

    elapsed = time.time() - start
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n[TEST] Accuracy = {test_acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    metrics = {
        "model": "ResNet18 (ImageNet pretrained, fine-tuned)",
        "dataset": args.dataset,
        "test_accuracy": round(test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs": args.epochs,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "training_time_min": round(elapsed / 60, 2),
        "history": history,
    }
    metric_path = os.path.join(METRIC_DIR, f"resnet18_{args.dataset}.json")
    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[✓] Métricas salvas em {metric_path}")
    plot_curves(history, args.dataset)


if __name__ == "__main__":
    main()
