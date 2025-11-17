#==========================================#
# Title:  Image classification with CNN
# Author: Jaewoong Han
# Date:   2024-06-27
#==========================================#
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnn_network import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 100
learning_rate = 0.0002
num_epoch = 15

mnist_train = datasets.MNIST(root="../data/", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = datasets.MNIST(root="../data/", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=True)

model = CNN(batch_size=batch_size).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr =[]
for i in range(num_epoch):
    epoch_loss = 0
    corrects = 0
    total = 0
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y= label.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(x)
        
        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(output, 1)
        corrects += (predicted == y).sum().item()
        total += y.size(0)

    epoch_loss /= len(train_loader)
    epoch_acc = corrects / total * 100
    print(f"epoch: [{str(i+1).zfill(2)}/{num_epoch}], Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.5f}")

correct = 0
total = 0
model.eval()

with torch.no_grad():
    for image,label in test_loader:
        x, y = image.to(device), label.to(device)
        output = model.forward(x)

        _,output_index = torch.max(output,1)

        total += label.size(0)
        correct += (output_index == y).sum().float()

    print(f"Accuracy of Test Data: {100*correct/total:.5f}%")