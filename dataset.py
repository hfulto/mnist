import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64):
    transform = transforms.ToTensor()
    
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    return train_loader
