import torch.nn as nn

class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                # Flatten 28x28 → 784
            nn.Linear(784, 16),         # First hidden layer
            nn.ReLU(),
            nn.Linear(16, 16),          # Second hidden layer
            nn.ReLU(),
            nn.Linear(16, 10)           # Output layer (10 classes)
        )

    def forward(self, x):
        return self.model(x)