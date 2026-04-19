#model.py

import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        
        # Block 1: Initial feature extraction
        self.stage1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: Intermediate features
        self.stage2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: Deep features
        self.stage3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with optimized neuron count
        self.lin1 = nn.Linear(128 * 4 * 4, 256)
        self.lin2 = nn.Linear(256, 10)

    def forward(self, tensor):
        # Propagation through convolutional blocks
        tensor = self.pool(F.relu(self.bn1(self.stage1(tensor))))
        tensor = self.pool(F.relu(self.bn2(self.stage2(tensor))))
        tensor = self.pool(F.relu(self.bn3(self.stage3(tensor))))
        
        # Flattening and classification
        tensor = tensor.view(-1, 128 * 4 * 4)
        tensor = F.relu(self.lin1(tensor))
        tensor = self.lin2(tensor)
        
        return tensor
