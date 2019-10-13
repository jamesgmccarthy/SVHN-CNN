import os
import h5py
from create_data_files import ImageProcessor
from Model import load_data
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

data_dir = './Data/processed'


class Conv_model(nn.Module):
    def __init__(self):
        super(Conv_model, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48,
                      kernel_size=5, padding=2),
            nn.Relu(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=2),
            nn.Dropout(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64,
                      kernel_size=5, padding=2),
            nn.Relu(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=2),
            nn.Dropout(0.2)
        )

        self.hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, padding=2),
            nn.Relu(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=2),
            nn.Dropout(0.2)
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160,
                      kernel_size=5, padding=2),
            nn.Relu(),
            nn.BatchNorm2d(160),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=2),
            nn.Dropout(0.2)
        )

        self.hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192,
                      kernel_size=5, padding=2),
            nn.Relu(),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=2),
            nn.Dropout(0.2)
        )

        self.hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.Relu(),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=2),
            nn.Dropout(0.2)
        )

        self.hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.Relu(),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=2),
            nn.Dropout(0.2)
        )

        self.hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.Relu(),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=2),
            nn.Dropout(0.2)
        )

        self.hidden9 = nn.Sequential(
            nn.Linear(192*8*8, 3072),
            nn.Relu()
        )

        self.hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.Relu()
        )
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        # Flatten
        x = x.view(x.size(0), 192*7*7)
        x = self.hidden9(x)
        x = self.hidden10(x)

        digit1_logits = self._digit1(x)
        digit2_logits = self._digit2(x)
        digit3_logits = self._digit3(x)
        digit4_logits = self._digit4(x)
        digit5_logits = self._digit5(x)

        return digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits
