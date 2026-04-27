"""3-layer CNN for CIFAR-10. Standard Kaiming init. Used as the H2 baseline."""

import torch.nn as nn


class CNN3(nn.Module):
    """Conv(3->32) -> Conv(32->64) -> Conv(64->128) -> FC(2048->10).

    Each conv: 3x3 kernel, padding=1, ReLU, 2x2 max pool.
    After 3 pools on 32x32 input: 128 x 4 x 4 = 2048 features.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def weight_layers(self):
        """Yield (name, module) pairs for layers with quantizable weights."""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                yield name, module
