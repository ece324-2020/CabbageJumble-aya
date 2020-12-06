import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from confusion_matrix import confusion
from models.model import coin_classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = torch.load("model_noHT.pt")
#model = torch.load("model2.pt")
#model.to(device)
#data_location = "Validation_noHT"
data_location = "test"

model = coin_classifier(12)
model.load_state_dict(torch.load("model_state_dict_model2.pt"))
model.to(device)
model.eval()


f = open("Normalization_Info.txt","r")
norm_info = f.read()
f.close()
norm_info = norm_info.split()
norm_info = [float(i) for i in norm_info]
R_mean,G_mean,B_mean,R_std,G_std,B_std = norm_info






transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = [R_mean,G_mean,B_mean],std = [R_std,G_std,B_std])])
data = torchvision.datasets.ImageFolder(data_location, transform = transform)
dataloader = torch.utils.data.DataLoader(data, batch_size=1,shuffle = True)

pred_vs_GT = []

correct = 0
num = 0

for idx, i in enumerate(dataloader):
    in_data = i[0].to(device)
    #print(in_data.shape)
    truth = i[1].item()
    predict =  model(in_data.float())
    predict.to(device)
    arg_predict = torch.argmax(predict,1).item()
    pair = [arg_predict,truth]
    pred_vs_GT.append(pair)
    if arg_predict == truth:
        correct+=1
    num+=1


    #print(in_data.shape)
print(correct/num)

pred_vs_GT = np.array(pred_vs_GT)
labels, prediction = pred_vs_GT[:, 0], pred_vs_GT[:, 1]
r = confusion(label = labels, prediction = prediction, plot = True, plot3d = True, text = True, HT = True)
print()
print(r)
print()

#labels, prediction = arr[:, 0], arr[:, 1]
#confusion(labels, prediction, plot = True, plot3d = True, text = True, HT = False)


    
