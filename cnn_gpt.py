import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Sample CNN class (same as earlier, but split into parts)
class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x  # return feature maps

# Instantiate model and load example data
model = MiniCNN()

# Get one MNIST sample
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
img, _ = dataset[0]  # Get a sample image
img = img.unsqueeze(0)  # Add batch dimension: [1, 1, 28, 28]

# Forward pass to get feature maps
with torch.no_grad():
    feature_maps = model.forward_features(img)

# Plot the feature maps (32 channels)
num_features = feature_maps.shape[1]
plt.figure(figsize=(12, 6))
for i in range(min(num_features, 16)):  # Show first 16 filters
    plt.subplot(4, 4, i+1)
    plt.imshow(feature_maps[0, i].numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f'Filter {i}')
plt.tight_layout()
plt.show()