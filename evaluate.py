import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def show_predictions(model, device, visualize):

    # Load MNIST test dataset
    transform = transforms.ToTensor()
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    loader = DataLoader(test_set, batch_size=1000, shuffle=True)

    # Start model evaluation

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            if visualize:
                images, labels = images.to(device), labels.to(device)

                # Send tensor with 8 images through the model
                outputs = model(images)

                # Get the probabilities of each digit for all images
                probs = F.softmax(outputs, dim=1)

                # Get the predicted digit (highest confidence) for all images
                preds = torch.argmax(probs, dim=1)

                for i in range(images.size(0)):

                    # --- Get image, label, prediction, and probabilities ---
                    image = images[i].squeeze()
                    label = labels[i].item()
                    pred = preds[i].item()
                    pred_probs = probs[i].cpu().numpy() * 100

                    # --- Plot image ---
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image, cmap='gray')
                    plt.title(f"True: {label} | Pred: {pred}")
                    plt.axis('off')

                    # --- Plot confidence (bar chart) ---
                    plt.subplot(1, 2, 2)
                    plt.bar(range(10), pred_probs)
                    plt.xticks(range(10))
                    plt.title("Prediction Confidence")
                    plt.xlabel("Digit")
                    plt.ylabel("Probability (%)")

                    plt.tight_layout()
                    plt.show()

            else:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')