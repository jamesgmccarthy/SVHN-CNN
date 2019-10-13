import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms


def loss(digit1_loss, digit2_loss, digit3_loss, digit4_loss, digit5_loss, digit_labels):
    # calculate loss using cross entropy, then return sum of all losses
    digit1_ce = F.cross_entropy(digit1_loss, digit_labels[0])
    digit2_ce = F.cross_entropy(digit2_loss, digit_labels[1])
    digit3_ce = F.cross_entropy(digit3_loss, digit_labels[2])
    digit4_ce = F.cross_entropy(digit4_loss, digit_labels[3])
    digit5_ce = F.cross_entropy(digit5_loss, digit_labels[4])
    loss = digit1_ce + digit2_ce+digit3_ce+digit4_ce+digit5_ce
    return loss


def train()
