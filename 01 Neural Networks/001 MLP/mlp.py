#==========================================#
# Title:  Image classification with MLP
# Author: Jaewoong Han
# Date:   2024-06-28
#==========================================#
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10 # datasets have image file of number 0 to 9
epochs = 15

"""
Step1: load datasets
 Load the MNIST dataset, applying transformations to normalize and
 convert images to tensor format for training and testing.
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for images, labels in train_loader:
    print(f"Shape of images: {images.shape}") # (128, 1, 28, 28)
    print(f"Shape of labels: {labels.shape}") # 128
    break

"""
Step2: Define the Multi-Layer Perceptron (MLP) Network
 Defines a neural network with two hidden layers and dropout for regularization to prevent overfitting.
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512) # (input vector, output)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784) # (batch size, 28, 28) -> (batch size, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Net()

"""
Step3: Define Loss Function and Optimizer, and Train the Model
 Uses CrossEntropyLoss for multi-class classification and RMSprop as an optimizer.
 The model is trained over multiple epochs.
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters())

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{str(epoch+1).zfill(2)}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

"""
Step4: test the model
"""
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    examples = []
    num_examples = 16

    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if len(examples) < num_examples:
            for i in range(len(labels)):
                if len(examples) < num_examples:
                    examples.append((images[i], labels[i], predicted[i]))
                else:
                    break

    print(f'Test Accuracy of the model on the {total} test images: {100 * correct / total}%')

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, (image, label, prediction) in enumerate(examples):
        image = image.squeeze().numpy() * 0.5 + 0.5
        ax = axes[i // 4, i % 4]
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {label.item()}\nPred: {prediction.item()}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()