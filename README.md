# CNN — MNIST & CIFAR-10: CustomCNN vs ResNet18

Implementação de uma CNN customizada em PyTorch com comparação frente à ResNet18 pré-treinada no ImageNet. Avaliação nos datasets MNIST e CIFAR-10, com visualização dos kernels/ativações (desafio).

---

## Estrutura do Projeto

```
cnn-mnist-cifar10/
├── src/
│   ├── model.py                  # Arquitetura CustomCNN
│   ├── train.py                  # Treino/eval da CustomCNN
│   ├── pretrained_comparison.py  # Fine-tuning ResNet18
│   ├── compare_results.py        # Gráficos comparativos
│   └── visualize_kernels.py      # DESAFIO: filtros e feature maps
├── results/
│   ├── plots/                    # Todos os gráficos gerados
│   └── metrics/                  # JSON com métricas de cada experimento
├── checkpoints/                  # Melhores modelos (.pth)
├── data/                         # Datasets baixados automaticamente
├── requirements.txt
└── README.md
```

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Executando os Experimentos (ordem recomendada)

```bash
# 1. Treinar CustomCNN nos dois datasets
python src/train.py --dataset mnist   --epochs 20
python src/train.py --dataset cifar10 --epochs 30

# 2. Fine-tuning ResNet18 pré-treinada
python src/pretrained_comparison.py --dataset mnist   --epochs 10
python src/pretrained_comparison.py --dataset cifar10 --epochs 10

# 3. Gerar gráficos comparativos
python src/compare_results.py

# 4. DESAFIO — Visualizar kernels e ativações
python src/visualize_kernels.py --dataset mnist
python src/visualize_kernels.py --dataset cifar10
```

---

## Arquitetura da CustomCNN

```
Input → ConvBlock(32, pool) → ConvBlock(64, pool) → ConvBlock(128) →
        FC(512) → BN → ReLU → Dropout(0.5) →
        FC(128) → ReLU → Dropout(0.25) →
        FC(10)
```

Cada `ConvBlock` = `Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → [MaxPool]`

**Regularizações utilizadas:**
- BatchNormalization (após cada convolução)
- Dropout (p=0.5 e p=0.25 na cabeça FC)
- Weight Decay via AdamW (λ=1e-4)
- Label Smoothing (ε=0.1)
- Data Augmentation (RandomCrop, Flip, ColorJitter, Affine)
- Cosine Annealing LR Scheduler

---

## Rede Pré-treinada: ResNet18

- Pesos pré-treinados no ImageNet (1.2M imagens, 1000 classes)
- Estratégia: congelamento de todas as camadas exceto `layer4` + `fc`
- Apenas ~11% dos parâmetros são retreinados
- MNIST: imagens convertidas para 3 canais (Grayscale→RGB) e redimensionadas para 32×32

---

## Gráficos Gerados

| Arquivo | Conteúdo |
|---|---|
| `custom_cnn_*_curves.png` | Loss e Accuracy por época (CustomCNN) |
| `resnet18_*_curves.png` | Loss e Accuracy por época (ResNet18) |
| `custom_cnn_*_confusion.png` | Matriz de confusão (CustomCNN) |
| `comparison_bar.png` | Barras: Test Accuracy dos 4 experimentos |
| `comparison_curves_*.png` | Sobreposição das curvas de val |
| `kernels_layer1_*.png` | Filtros aprendidos na 1ª camada Conv |
| `feature_maps_block*_*.png` | Ativações por bloco convolucional |
| `activation_overview_*.png` | Imagem original + heatmaps de ativação |
