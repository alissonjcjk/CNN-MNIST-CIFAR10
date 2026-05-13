"""
run_colab.py
============
Execute no Google Colab como:

    !python run_colab.py

Ou célula por célula, copiando cada seção marcada com # === CÉLULA ===
"""

import os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n{'─'*55}")
    subprocess.run(cmd, check=True, cwd=ROOT)

# ── ETAPA 1: Verifica GPU ──────────────────────────────────
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (sem GPU)"
print(f"Dispositivo: {gpu}")

# ── ETAPA 2: CustomCNN ────────────────────────────────────
run(["python", "src/train.py", "--dataset", "mnist",   "--epochs", "20"])
run(["python", "src/train.py", "--dataset", "cifar10", "--epochs", "30"])

# ── ETAPA 3: ResNet18 pré-treinada ────────────────────────
run(["python", "src/pretrained_comparison.py", "--dataset", "mnist",   "--epochs", "10"])
run(["python", "src/pretrained_comparison.py", "--dataset", "cifar10", "--epochs", "10"])

# ── ETAPA 4: Gráficos comparativos ────────────────────────
run(["python", "src/compare_results.py"])

# ── ETAPA 5: Visualização de kernels (DESAFIO) ────────────
run(["python", "src/visualize_kernels.py", "--dataset", "mnist"])
run(["python", "src/visualize_kernels.py", "--dataset", "cifar10"])

# ── ETAPA 6: Mostra resultados inline ─────────────────────
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

PLOT_DIR = Path(ROOT) / "results" / "plots"
plots = sorted(PLOT_DIR.glob("*.png"))

print(f"\n{'='*55}")
print(f"  {len(plots)} gráficos gerados em results/plots/")
print(f"{'='*55}")

cols = 2
rows = (len(plots) + 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
axes = axes.flat

for ax, p in zip(axes, plots):
    img = mpimg.imread(str(p))
    ax.imshow(img)
    ax.set_title(p.stem, fontsize=9)
    ax.axis("off")

for ax in list(axes)[len(plots):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(str(PLOT_DIR / "_todos_os_graficos.png"), dpi=100)
plt.show()
print("\n[✓] Todos os experimentos concluídos!")
