# model.py

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ChickenResNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Load ResNet18 without pretrained weights
        self.backbone = resnet18(weights=None)

        # Modify the first convolution layer to accept 1-channel input
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Replace the final FC layer for our 3 classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_model(num_classes: int = 3):
    """
    Returns:
        model: ChickenResNet
        device: 'cuda' or 'cpu'
    """
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ChickenResNet(num_classes=num_classes)

    # Move model to the detected device
    model = model.to(device)

    return model, device


if __name__ == "__main__":
    print("Testing ChickenResNet...")

    model, device = get_model(num_classes=3)
    print("Using device:", device)

    # dummy mel-spectrogram: [batch, channels, n_mels, time]
    dummy = torch.randn(4, 1, 64, 94).to(device)

    out = model(dummy)
    print("Output shape:", out.shape)
