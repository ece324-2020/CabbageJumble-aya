import torch
import torch.utils.data as dt
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def load_data(batch_size, DIR):
    # Get the loader
    transform1 = transforms.Compose([transforms.ToTensor()])
    #trainset = torchvision.datasets.ImageFolder(DIR,transform=transform)

    # transform
    trainset1 = torchvision.datasets.ImageFolder(DIR, transform=transform1)
    trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=len(trainset1),
                                         shuffle=False, num_workers=2)
    return trainloader1

def mean_std(trainloader):
    imgs = 0
    mean = 0.0
    variance = 0.0
    for i, batch in enumerate(trainloader1):
        # Get inputs
        inputs = batch[0]
        # reshape the batch
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        # Get sum of length of images
        imgs = imgs + inputs.size(0)
        # Calculate mean and variance for each image
        mean = mean + inputs.mean(2).sum(0) 
        variance = variance+ inputs.var(2).sum(0)
    # Calculate whole mean and std
    mean = mean/imgs
    variance = variance/imgs
    # Get the standard deviation by squareroot of variance
    std = torch.sqrt(variance)
    return tuple(mean.tolist()), tuple(std.tolist())

class dataset(dt.Dataset):
    def __init__(self, X, y, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean1, std1)]):
        self.X = X
        self.y = y
        self.transform = transform
        self.sample = {}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X_tensor = self.transform(self.X[index])
        y_tensor = transform.ToTensor(self.y[index])
        # returns tensors input and label
        #arr_np = np.array([self.X[index]]).squeeze()
        #arr_np_label = np.array([self.y[index]])
        return X_tensor, y_tensor
