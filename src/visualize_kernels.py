"""
visualize_kernels.py
====================
DESAFIO: Visualização das ativações dos kernels/filtros da CustomCNN.

Gera dois tipos de visualização:
  1. Pesos dos filtros da primeira camada convolucional
  2. Feature maps (ativações) para uma imagem de exemplo

Uso:
    python src/visualize_kernels.py --dataset cifar10
    python src/visualize_kernels.py --dataset mnist
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from model import CustomCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_model(dataset_name: str) -> CustomCNN:
    """Carrega o melhor checkpoint da CustomCNN."""
    in_ch = 1 if dataset_name == "mnist" else 3
    sz = 28 if dataset_name == "mnist" else 32
    model = CustomCNN(in_channels=in_ch, num_classes=10,
                      input_size=sz, dropout_rate=0.5)
    ckpt_path = os.path.join(CKPT_DIR, f"custom_cnn_{dataset_name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[AVISO] Checkpoint não encontrado em {ckpt_path}")
        print("         Execute train.py primeiro. Usando pesos aleatórios.")
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"[✓] Pesos carregados de {ckpt_path}")
    model.to(DEVICE).eval()
    return model


def get_sample_image(dataset_name: str):
    """Retorna uma imagem de amostra do dataset."""
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    img, label = ds[0]
    return img.unsqueeze(0), label, ds


# ──────────────────────────────────────────────────────────────────────────────
# 1. Pesos dos kernels (1ª camada Conv)
# ──────────────────────────────────────────────────────────────────────────────
def plot_first_layer_filters(model: CustomCNN, dataset_name: str):
    """
    Plota os filtros aprendidos pela primeira camada convolucional.
    Para MNIST (1 canal) os filtros têm shape (32, 1, 3, 3).
    Para CIFAR-10 (3 canais) os filtros têm shape (32, 3, 3, 3) → mostramos magnitude.
    """
    # Acessa primeiro Conv2d no primeiro ConvBlock
    first_conv = list(model.features[0].block.children())[0]  # nn.Conv2d
    kernels = first_conv.weight.data.cpu()  # (out_channels, in_ch, kH, kW)

    n_filters = kernels.shape[0]
    cols = 8
    rows = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"Filtros da 1ª Camada Conv — {dataset_name.upper()}", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            k = kernels[i]
            if k.shape[0] == 1:
                img = k[0].numpy()
                ax.imshow(img, cmap="viridis")
            else:
                # Normaliza para [0,1] para visualização como RGB
                img = k.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"kernels_layer1_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Filtros da 1ª camada salvo em {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature Maps (ativações) por bloco
# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_maps(model: CustomCNN, img_tensor: torch.Tensor, dataset_name: str):
    """
    Plota os feature maps gerados por cada bloco convolucional
    para a imagem de entrada fornecida.
    """
    img_tensor = img_tensor.to(DEVICE)
    with torch.no_grad():
        feature_maps = model.get_feature_maps(img_tensor)

    block_names = ["Bloco 1 (32 canais, /2)", "Bloco 2 (64 canais, /4)", "Bloco 3 (128 canais)"]

    for block_idx, (fmap, name) in enumerate(zip(feature_maps, block_names)):
        fmap = fmap.squeeze(0).cpu()  # (C, H, W)
        n_channels = min(fmap.shape[0], 32)  # mostramos até 32 canais
        cols = 8
        rows = (n_channels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle(f"Feature Maps — {name} — {dataset_name.upper()}", fontsize=13)

        for i, ax in enumerate(axes.flat):
            if i < n_channels:
                act = fmap[i].numpy()
                ax.imshow(act, cmap="inferno")
                ax.axis("off")
                ax.set_title(f"F{i}", fontsize=7)
            else:
                ax.axis("off")

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"feature_maps_block{block_idx+1}_{dataset_name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  [✓] Feature maps Bloco {block_idx+1} salvos em {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Imagem original vs ativações (visão geral)
# ──────────────────────────────────────────────────────────────────────────────
def plot_activation_overview(model: CustomCNN, img_tensor: torch.Tensor,
                              label: int, dataset_name: str, ds):
    """
    Painel: imagem original + ativação média por bloco (pseudo-heatmap).
    """
    # Desnormaliza a imagem para exibição
    if dataset_name == "mnist":
        mean = np.array([0.1307])
        std = np.array([0.3081])
        raw_img = img_tensor.squeeze(0).squeeze(0).cpu().numpy()
        raw_img = raw_img * std[0] + mean[0]
        raw_img = np.clip(raw_img, 0, 1)
    else:
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        raw_img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        raw_img = raw_img * std + mean
        raw_img = np.clip(raw_img, 0, 1)

    img_tensor = img_tensor.to(DEVICE)
    with torch.no_grad():
        feature_maps = model.get_feature_maps(img_tensor)

    class_names_mnist = [str(i) for i in range(10)]
    class_names_cifar = ["airplane", "automobile", "bird", "cat", "deer",
                         "dog", "frog", "horse", "ship", "truck"]
    cnames = class_names_mnist if dataset_name == "mnist" else class_names_cifar

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Ativações por Bloco — Classe: '{cnames[label]}' — {dataset_name.upper()}",
        fontsize=13
    )

    # Imagem original
    if dataset_name == "mnist":
        axes[0].imshow(raw_img, cmap="gray")
    else:
        axes[0].imshow(raw_img)
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")

    # Ativação média de cada bloco
    block_names = ["Bloco 1\n(32 ch, /2)", "Bloco 2\n(64 ch, /4)", "Bloco 3\n(128 ch)"]
    for i, (fmap, bname) in enumerate(zip(feature_maps, block_names)):
        act_mean = fmap.squeeze(0).mean(0).cpu().numpy()
        # Upsample para o tamanho original para facilitar comparação visual
        from PIL import Image as PILImage
        act_img = PILImage.fromarray(
            ((act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8) * 255).astype(np.uint8)
        )
        orig_size = 28 if dataset_name == "mnist" else 32
        act_img = act_img.resize((orig_size, orig_size), PILImage.BILINEAR)
        axes[i + 1].imshow(np.array(act_img), cmap="inferno")
        axes[i + 1].set_title(bname)
        axes[i + 1].axis("off")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"activation_overview_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Painel de ativações salvo em {path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["mnist", "cifar10"])
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Visualização de Kernels/Ativações — {args.dataset.upper()}")
    print(f"{'='*55}\n")

    model = load_model(args.dataset)
    img_tensor, label, ds = get_sample_image(args.dataset)

    print("\n[1/3] Plotando filtros da 1ª camada convolucional...")
    plot_first_layer_filters(model, args.dataset)

    print("\n[2/3] Plotando feature maps por bloco...")
    plot_feature_maps(model, img_tensor, args.dataset)

    print("\n[3/3] Plotando painel geral de ativações...")
    plot_activation_overview(model, img_tensor, label, args.dataset, ds)

    print("\n[✓] Todas as visualizações concluídas!")
    print(f"    Salvas em: results/plots/")


if __name__ == "__main__":
    main()
