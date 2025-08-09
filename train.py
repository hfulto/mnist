import torch
import torch.nn as nn
import torch.optim as optim

from model import BasicMLP
from dataset import get_mnist_loaders
from evaluate import show_predictions

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
train_loader = get_mnist_loaders()

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} complete - Loss: {loss.item():.4f}")

    show_predictions(model, device, visualize=False)

# show_predictions(model, device, visualize=True)