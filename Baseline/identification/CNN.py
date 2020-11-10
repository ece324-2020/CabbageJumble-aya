import torch
import torch.nn as nn
import torch.nn.functional as F


# For now
class CNN_baseline(nn.Module):
    def __init__(self):
        super(CNN_baseline, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        
    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        return


