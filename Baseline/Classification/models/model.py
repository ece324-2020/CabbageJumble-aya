import torch
import torch.nn as nn
import torch.nn.functional as F

class coin_classifier(nn.Module):
    def __init__(self,num_classes):
        super(coin_classifier,self).__init__()
        #input size is 100 x 100 x 3
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 10,kernel_size = 3, stride = 1)
        #size is 98 x 98 x 10
        self.pool = nn.MaxPool2d(2, stride = 2)
        #size is 49x49 x 10
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20,kernel_size = 3, stride = 1)
        #size is 47x47 x 20
        #max pool --> 23x23 x 20
    
        self.conv3 = nn.Conv2d(in_channels = 20, out_channels = 40,kernel_size = 3, stride = 1)
        #size is 21 x 21 x 40
        # max pool --> 10 x 10 x 40
        self.conv4 = nn.Conv2d(in_channels = 40, out_channels = 80,kernel_size = 3, stride = 1)
        #size is 8 x 8 x 80
        #max pool --> 4x4x80
        
        self.fc1 = nn.Linear(4*4*80,64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self,in_batch):
        x = in_batch
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1,4*4*80)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x)
        x= self.softmax(x)
        #print(x)
        return x
        