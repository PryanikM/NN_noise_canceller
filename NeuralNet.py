import torch.nn as nn
from Hyperparams import Hyperparams as hp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DenoisingNet(nn.Module):
    def __init__(self):
        super(DenoisingNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)  # Выходной канал также 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class AudioDenoisingModel(nn.Module):
    def __init__(self, inputs, level_1, n_levels, levels_repeat, dropout_rate=0.0):
        super(AudioDenoisingModel, self).__init__()

        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(nn.Conv1d(in_channels=inputs[1], out_channels=level_1, kernel_size=3, padding=1))
        if dropout_rate > 0:
            self.layers.append(nn.Dropout(p=dropout_rate))

        # Intermediate levels
        for j in range(n_levels):
            for _ in range(levels_repeat):
                out_channels = 2 ** (n_levels - j)
                self.layers.append(
                    nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(p=dropout_rate))

        # Output layer
        self.layers.append(nn.Conv1d(in_channels=2 ** n_levels, out_channels=1, kernel_size=3, padding=1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
