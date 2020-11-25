import torch

import torch.nn as nn



class coin_classifier(nn.Module):

    def __init__(self):

        #input size is 100

        self.conv1 = nn.Conv2d(3,15,3)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(15,30,3)



    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        



