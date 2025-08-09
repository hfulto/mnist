
## Handwritten digit recognizer (MNIST)
Starting from basics with deep learning by making a Multilayer Perceptron (MLP) model to recognize handwritten digits using the public MNIST dataset with PyTorch

After it's trained the code will show a test digit image along with the model's prediction for it and confidence

### Model design

Stolen from 3blue1brown

- Input layer: 28x28 pixel greyscale image (784 neurons once flattened)
- Hidden layer 1: 16 neurons
- Hidden layer 2: 16 neurons
- Output layer: 10 neurons (1 for each digit)

Activation function: ReLU

Optimizer (training algorithm): Stochastic Gradient Descent (SGD)

Cost function (aka loss): Cross Entropy

- Different to 3blue1brown, he uses Mean Squared Error (MSE)

### Setup

You will need to install PyTorch and the torchvision libraries to run it \
Ensure you also have matplotlib and numpy

`pip install torch torchvision matplotlib numpy`

### Run it

```
cd mnist
python ./train.py
```
