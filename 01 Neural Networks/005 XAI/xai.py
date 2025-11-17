#==========================================#
# Title:  Image classification with CAM
# Author: Jaewoong Han
# Date:   2024-06-25
# Reference code on
#  - https://poddeeplearning.readthedocs.io/ko/latest/CNN/VGG19%20+%20GAP%20+%20CAM/
#  - https://junstar92.tistory.com/152
#==========================================#
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10, ImageNet, STL10
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import skimage.transform

num_epochs = 15

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = STL10(root='../data', split='train', download=True, transform=transform)
test_dataset = STL10(root='../data', split='test', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Dataset size: {len(train_loader)}")
image, label = next(iter(train_loader))
print(f"Image shape: {image.shape}")

base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
base_model.classifier = nn.Identity()

class VGG16_with_GAP(nn.Module):
    def __init__(self):
        super(VGG16_with_GAP, self).__init__()
        self.base = base_model.features
        self.gap = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

        for param in list(self.base.parameters())[:-4]:
            param.requires_grad = False

    def forward(self, x):
        x = self.base(x)
        features = x.clone()
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x, features

model = VGG16_with_GAP()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(),lr=0.00001)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset) * 100

    print(f'Epoch {str(epoch+1).zfill(2)}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f} %')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

for data in test_loader:
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    outputs, features = model(images)
    _, predicted = torch.max(outputs, 1)
    break

classes =  ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
params = list(model.parameters())[-2]
num = 0
for num in range(64):
    print(f"Item {str(num+1).zfill(2)}/64 | Ground Truth: {classes[int(labels[num])]} | Prediction: {classes[int(predicted[num])]}")

    overlay = params[int(predicted[num])].matmul(features[num].reshape(512,49)).reshape(7,7).cpu().data.numpy()

    overlay = overlay - np.min(overlay)
    overlay = overlay / np.max(overlay)
    overlay_resized = skimage.transform.resize(overlay, [224, 224])

    original_image = images[num].cpu()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    img = original_image.permute(1, 2, 0).numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(img)
    ax[1].imshow(overlay_resized, alpha=0.4, cmap='jet')
    ax[1].set_title("Learned Overlay")
    ax[1].axis('off')

    plt.show()