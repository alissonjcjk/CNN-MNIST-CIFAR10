"""
model.py
========
CNN customizada com regularização (BatchNorm + Dropout).

Arquitetura inspirada em VGG: blocos convolucionais com kernels 3x3,
MaxPooling para redução espacial, e classificador FC no topo.

Adaptável a:
  - MNIST  (1 canal, 10 classes, 28x28)
  - CIFAR-10 (3 canais, 10 classes, 32x32)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloco convolucional: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    CNN customizada com dois blocos convolucionais e cabeça FC.

    Parâmetros
    ----------
    in_channels : int
        1 para MNIST, 3 para CIFAR-10.
    num_classes : int
        Número de classes de saída (10 para ambos os datasets).
    input_size : int
        Dimensão espacial da entrada (28 para MNIST, 32 para CIFAR-10).
    dropout_rate : float
        Taxa de Dropout na cabeça FC (regularização).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, pool=True),   # -> (32, H/2, W/2)
            ConvBlock(32, 64, pool=True),             # -> (64, H/4, W/4)
            ConvBlock(64, 128, pool=False),           # mantém resolução
        )

        # Calcula o tamanho do mapa de features após os pools
        feat_size = input_size // 4  # 2 MaxPool2d(2,2)
        flat_size = 128 * feat_size * feat_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes),
        )

        # Inicialização de pesos (He/Kaiming para ReLU)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_feature_maps(self, x: torch.Tensor) -> list:
        """Retorna os feature maps intermediários (para visualização de ativações)."""
        maps = []
        for layer in self.features:
            x = layer(x)
            maps.append(x)
        return maps
