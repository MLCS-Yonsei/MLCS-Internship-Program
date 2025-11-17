#==========================================#
# Title:  Test convolution operation
# Author: Jaewoong Han
# kernel information on: https://en.wikipedia.org/wiki/Kernel_(image_processing)
# Date:   2024-06-28
#==========================================#
import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F

def ReLU(x):
    return np.maximum(0, x)

transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# If you want to apply in more high-resolution image, use this.
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor()
# ])
# train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)

image, label = train_dataset[6]

#define filter
filter_vertical = np.array([
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0]
])
filter_sobel = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
])
filter_gaussian_blur = np.array([
    [1.0, 2.0, 1.0], 
    [2.0, 4.0, 2.0], 
    [1.0, 2.0, 1.0]
]) / 16

filter_choosen = filter_vertical

image_tensor = image.unsqueeze(0)
filter_tensor = torch.tensor(filter_choosen, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
convolved_image = F.conv2d(image_tensor, filter_tensor, padding=1)


fig, axes = plt.subplots(4, 3, figsize=(12, 12))
axes[0, 0].imshow(image.squeeze(), cmap='gray')
axes[0, 0].set_title(f'Original Image')
axes[0, 0].axis('on')

axes[0, 1].imshow(filter_choosen, cmap='gray')
axes[0, 1].set_title('Filter')
axes[0, 1].axis('on')


axes[0, 2].imshow(convolved_image.squeeze().detach().numpy(), cmap='gray')
axes[0, 2].imshow(image.squeeze(), cmap='gray', alpha=0.8)
axes[0, 2].set_title('Convolved Image')
axes[0, 2].axis('on')

for i in range(3):
    for j in range(3):
        convolved_image = F.conv2d(convolved_image, filter_tensor, padding=1) # no padding in this loop
        convolved_image = ReLU(convolved_image.squeeze()).unsqueeze(0).unsqueeze(0)
        axes[i + 1, j].imshow(convolved_image.squeeze().detach().numpy(), cmap='gray')
        axes[i + 1, j].set_title('Convolved Image')
        axes[i + 1, j].axis('off')

plt.show()