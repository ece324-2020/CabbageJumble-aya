import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from torchsummary import summary

#example to run the code
#py Sec3_main.py --data_location training/my_training/ --seed 0 --lr 0.01 --epochs 100 --batch_size 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#getting required arguements
parser = argparse.ArgumentParser()
parser.add_argument('--images_path',required = True)
parser.add_argument('--labels_path',required = True)
parser.add_argument('--seed', type=int, default = 1)
parser.add_argument('--lr', type=float, default = 0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default = 4)

args = parser.parse_args()

#setting random seed
torch.manual_seed(args.seed)

#loading unnormalized data
temp_transform = transforms.Compose([transforms.ToTensor()])
train_temp_data = torchvision.datasets.ImageFolder(args.data_location, transform = temp_transform)
temp_loader = torch.utils.data.DataLoader(train_temp_data, batch_size=len(train_temp_data))

#getting the mean and std of the data
data = iter(temp_loader)
data = data.next()

R = (data[0])[:,0,:,:]
G = (data[0])[:,1,:,:]
B = (data[0])[:,2,:,:]

R_mean = torch.mean(R)
G_mean = torch.mean(G)
B_mean = torch.mean(B)

R_std = torch.std(R)
G_std = torch.std(G)
B_std = torch.std(B)

#normalizing and loading the data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = [R_mean,G_mean,B_mean],std = [R_std,G_std,B_std])])
data = torchvision.datasets.ImageFolder(args.data_location, transform = transform)
dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,shuffle=True)



#getting model, loss function, and optimizer
model = coin_classifier()
model.to(device)


loss_fnc = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)

#keeping track of loss, accuracy, and time
train_loss = []
train_acc = []
start_time = time.time()

#maing training loop
for epoch in range(args.epochs):
    running_loss = 0
    num_batches = 0
    #iterating through mini-batches
    for idx, i in enumerate(dataloader):
        optimizer.zero_grad()
        #potential error
        in_data = i[0].to(device)
        truth = i[1].to(device)

        #manually onehot encoding truth (the labels)
        new_labels = torch.zeros((len(in_data),10))
        for idx2,j in enumerate(truth):
            new_labels[idx2,j]=1
        
        predict =  model(in_data.float())
        loss = loss_fnc(input=predict, target=new_labels.float())
        running_loss += loss
        num_batches += 1
        loss.backward()
        optimizer.step()    
    #this is for average loss per epoch
    train_loss.append(running_loss/num_batches)
    num_correct = 0
    
    #getting the accuracy per an epoch
    for idx,i in enumerate(dataloader):
        in_data = i[0]
        truth = i[1]
        #manually onehot encoding the labels
        new_labels = torch.zeros((len(in_data),10))
        for idx2,j in enumerate(truth):
            new_labels[idx2,j]=1
        predict =  model(in_data.float())
        predict = torch.max(predict,1)

        #getting the number of correct predictions
        for k in range(truth.size()[0]):
            if truth[k] == predict[1][k]:
                num_correct+=1
    
    train_acc.append(num_correct/len(data))

end_time = time.time()

print(f"Time took is: {end_time-start_time}")


#plotting 
x = list(range(len(train_acc)))
train_acc = [i*100 for i in train_acc]

#plotting the accuracy per epoch
plt.plot(x, train_acc, label = "train_acc")
plt.xlabel('Epoch number')
plt.ylabel("Accuracy (in percentage)")
plt.title(f"Training Accurarcy: alpha = {args.lr}, epoch_num = {args.epochs}, batch_size = {args.batch_size}")
plt.legend(loc='lower right')
plt.show()

#plotting the loss per epoch
plt.plot(x, train_loss, label = "train_loss")
plt.xlabel('Epoch number')
plt.ylabel("Accuracy (in percentage)")
plt.title(f"Training Loss: alpha = {args.lr}, epoch_num = {args.epochs}, batch_size = {args.batch_size}")
plt.legend(loc='upper right')
plt.show()

#displaying model statistics
summary(model, input_size=(3, 56, 56))
