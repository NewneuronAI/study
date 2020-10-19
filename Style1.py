import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

tf = transforms.Compose([

    transforms.ToTensor,
    transforms.Normalize
])