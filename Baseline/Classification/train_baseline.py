import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from models.model import *

#example to run the code
#py train_baseline.py --data_location DATA --num_classes 4 --val_data_location Validation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#getting required arguements
parser = argparse.ArgumentParser()
parser.add_argument('--data_location',required = True)
parser.add_argument('--val_data_location',required = True)
parser.add_argument('--seed', type=int, default = 1)
parser.add_argument('--lr', type=float, default = 0.1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default = 20)
parser.add_argument('--num_classes', type=int, default = 12)
parser.add_argument('--save_name', type=str, default = "model3.pt")




args = parser.parse_args()

#setting random seed
torch.manual_seed(args.seed)


def accuracy_and_loss(dataloader,length_of_loader,model,loss_fnc,device):
    num_correct = 0
    running_loss = 0
    num_iter = 0
    for idx,i in enumerate(dataloader):
        in_data = i[0].to(device)
        truth = i[1].to(device)
        gt_labels = torch.zeros(truth.shape[0],args.num_classes)
        for idx2, j in enumerate(truth):
            gt_labels[idx2][j] = 1

        
        gt_labels = gt_labels.to(device)
        predict =  model(in_data.float())
        predict = predict.to(device)

        loss = loss_fnc(input=predict, target=gt_labels.float())
        running_loss += float(loss)
        num_iter+=1
        
        arg_predict = torch.argmax(predict,1)
        #getting the number of correct predictions
        for k in range(truth.size()[0]):
            if truth[k] == arg_predict[k]:
                num_correct+=1
    return num_correct*100/length_of_loader, running_loss/num_iter




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

f = open("Normalization_Info2.txt","a")
f.write(f"{R_mean} {G_mean} {B_mean} {R_std} {G_std} {B_std}")
f.close()


val_data = torchvision.datasets.ImageFolder(args.val_data_location, transform = transform)
valloader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data),shuffle=True)

print("Done Loading Data")

#getting model, loss function, and optimizer
model = coin_classifier(args.num_classes)
model.to(device)
#model = torch.load("model.pt")


loss_fnc = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)

#keeping track of loss, accuracy, and time
train_loss = []
train_acc = []
val_loss = []
val_acc = []
start_time = time.time()

#maing training loop
for epoch in range(args.epochs):
    print(f"Epoch num: {epoch}")
    #running_loss = 0
    #num_batches = 0
    #iterating through mini-batches
    for idx, i in enumerate(dataloader):
        optimizer.zero_grad()
        
        in_data = i[0].to(device)
        truth = i[1].to(device)
        gt_labels = torch.zeros(truth.shape[0],args.num_classes).to(device)
        for idx2, j in enumerate(truth):
            gt_labels[idx2][j] = 1
    
        predict =  model(in_data.float())
        predict.to(device)
        loss = loss_fnc(input=predict, target=gt_labels.float())
        #running_loss += float(loss)
        #num_batches += 1
        loss.backward()
        optimizer.step()    
    #this is for average loss per epoch
    #train_loss.append(running_loss/num_batches)
    #num_correct = 0
    '''
    #getting the train accuracy per an epoch
    for idx,i in enumerate(dataloader):
        in_data = i[0].to(device)
        truth = i[1].to(device)
        gt_labels = torch.zeros(truth.shape[0],args.num_classes)
        for idx2, j in enumerate(truth):
            gt_labels[idx2][j] = 1
        gt_labels.to(device)
        predict =  model(in_data.float())
        predict.to(device)
        
        arg_predict = torch.argmax(predict,1)
        #getting the number of correct predictions
        for k in range(truth.size()[0]):
            if truth[k] == arg_predict[k]:
                num_correct+=1
    '''
    training_acc,training_loss = accuracy_and_loss(dataloader,len(data),model,loss_fnc,device)
    train_acc.append(training_acc)
    train_loss.append(training_loss)

    #validation accuracy
    val_accuracy, val_loss_value = accuracy_and_loss(valloader,len(val_data),model,loss_fnc,device)
    val_acc.append(val_accuracy)
    val_loss.append(val_loss_value)
    
    #train_acc.append(num_correct/len(data))

end_time = time.time()

print(f"Time took is: {end_time-start_time}")


#plotting 
x = list(range(len(train_acc)))

#plotting the accuracy per epoch
plt.plot(x, train_acc, label = "train_acc")
plt.plot(x,val_acc,label = "val_acc")
plt.xlabel('Epoch number')
plt.ylabel("Accuracy (in percentage)")
plt.title(f"Training Accurarcy: alpha = {args.lr}, epoch_num = {args.epochs}, batch_size = {args.batch_size}")
plt.legend(loc='lower right')
plt.show()

#plotting the loss per epoch
plt.plot(x, train_loss, label = "train_loss")
plt.plot(x, val_loss, label = "val_loss")
plt.xlabel('Epoch number')
plt.ylabel("Loss")
plt.title(f"Training Loss: alpha = {args.lr}, epoch_num = {args.epochs}, batch_size = {args.batch_size}")
plt.legend(loc='upper right')
plt.show()

print(f"Train Accuracy is: {train_acc[-1]}")
print(f"Validation Accuracy is: {val_acc[-1]}")

#displaying model statistics
#summary(model, input_size=(3, 56, 56))



torch.save(model,args.save_name)
