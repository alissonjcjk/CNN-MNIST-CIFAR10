"""
compare_results.py
==================
Lê os arquivos JSON de métricas gerados pelo train.py e pretrained_comparison.py
e gera gráficos comparativos consolidados.

Uso (após rodar os treinamentos):
    python src/compare_results.py
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

METRIC_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")
PLOT_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_metric(filename: str) -> dict | None:
    path = os.path.join(METRIC_DIR, filename)
    if not os.path.exists(path):
        print(f"[AVISO] Arquivo não encontrado: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def bar_comparison():
    """Gráfico de barras: Test Accuracy dos 4 experimentos."""
    experiments = {
        "CustomCNN\nMNIST":    load_metric("custom_cnn_mnist.json"),
        "ResNet18\nMNIST":     load_metric("resnet18_mnist.json"),
        "CustomCNN\nCIFAR-10": load_metric("custom_cnn_cifar10.json"),
        "ResNet18\nCIFAR-10":  load_metric("resnet18_cifar10.json"),
    }

    labels, accs, colors = [], [], []
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for (label, data), color in zip(experiments.items(), palette):
        if data:
            labels.append(label)
            accs.append(data["test_accuracy"] * 100)
            colors.append(color)

    if not labels:
        print("[AVISO] Nenhuma métrica encontrada. Execute os treinamentos primeiro.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accs, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Comparação de Desempenho: CustomCNN vs ResNet18 Pré-Treinada", fontsize=13)
    ax.axhline(y=99, color="gray", linestyle="--", linewidth=0.8, label="99% referência")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "comparison_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Gráfico de barras salvo em {path}")


def learning_curve_overlay(dataset_name: str):
    """Sobreposição das curvas de validação: CustomCNN vs ResNet18."""
    custom = load_metric(f"custom_cnn_{dataset_name}.json")
    resnet = load_metric(f"resnet18_{dataset_name}.json")

    if not custom and not resnet:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Curvas de Aprendizado — {dataset_name.upper()}", fontsize=13)

    for ax, key, ylabel in zip(axes, ["val_loss", "val_acc"], ["Val Loss", "Val Accuracy"]):
        if custom:
            ax.plot(custom["history"][key], label="CustomCNN", color="#4C72B0", linewidth=2)
        if resnet:
            ax.plot(resnet["history"][key], label="ResNet18", color="#C44E52",
                    linewidth=2, linestyle="--")
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"comparison_curves_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Curvas comparativas ({dataset_name}) salvas em {path}")


def summary_table():
    """Imprime tabela resumo no terminal."""
    files = {
        "CustomCNN — MNIST":     "custom_cnn_mnist.json",
        "ResNet18  — MNIST":     "resnet18_mnist.json",
        "CustomCNN — CIFAR-10":  "custom_cnn_cifar10.json",
        "ResNet18  — CIFAR-10":  "resnet18_cifar10.json",
    }
    print("\n" + "="*70)
    print(f"{'Modelo':<25} {'Test Acc':>10} {'Val Acc':>10} {'Épocas':>8} {'Tempo(min)':>12}")
    print("-"*70)
    for label, fname in files.items():
        data = load_metric(fname)
        if data:
            print(f"{label:<25} {data['test_accuracy']*100:>9.2f}% "
                  f"{data['best_val_accuracy']*100:>9.2f}% "
                  f"{data['epochs']:>8} "
                  f"{data['training_time_min']:>11.1f}")
        else:
            print(f"{label:<25} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>12}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n[Gerando comparações visuais...]\n")
    bar_comparison()
    learning_curve_overlay("mnist")
    learning_curve_overlay("cifar10")
    summary_table()
    print("[✓] Comparação concluída!")
