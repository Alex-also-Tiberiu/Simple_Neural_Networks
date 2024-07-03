import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Simple single layer feed-forward network.
# A fully connected layer is initialized with the utility method torch.nn.Linear.
# Keep in mind that the activation function is not included!
class Net(nn.Module):

    # This is exectuted when the object is initialized (no need to call it explicitly).
    # Here you have to instantiate all the network's parameters.
    # PyTorch provides utility functions to easily initialize most of the commonly used deep learning layers.
    def __init__(self, Ni, Nh, No):
        """
        Ni - Input size
        Nh - Neurons in the hidden layer
        No - Output size
        """
        super().__init__()

        print('Network initialized')
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh)
        self.out = nn.Linear(in_features=Nh, out_features=No)
        self.act = nn.Sigmoid()

    # Here you define the forward pass of the network, from the input x to the output (the method must return the network output).
    # You just need to define the forward part, the back-propagation is automatically tracked by the framework!
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.out(x)
        return x

# For reproducibility, it is always recommended to set a manual seed.
# In this way the randomly initialized network's parameters will be always the same.
# Try to disable it to see how the initialized weights change every time you redefine the network object.
# Network parameters
Ni = 1
Nh = 32
No = 1

torch.manual_seed(0)
net = Net(Ni, Nh, No)

# Print network structure
print(net)

x = torch.rand(128, 1)
out = net(x)
print(f"OUTPUT VALUES: {out}")
print(f"OUTPUT SHAPE: {out.shape}")

# Disable gradient
with torch.no_grad():
    out = net(x)
print(f"OUTPUT SHAPE: {out.shape}")

# Run on GPU
### Check if a cuda GPU is available
if torch.cuda.is_available():
    print('GPU availble')
    # Define the device (here you can select which GPU to use if more than 1)
    device = torch.device("cuda")
else:
    print('GPU not availble')
    device = torch.device("cpu")

print(f"SELECTED DEVICE: {device}")

### Transfer the network parameters to the GPU memory (if available)
net.to(device)

### Transfer the input data to the GPU memory (if available) and compute output
out = net(x.to(device))

print(f"OUTPUT SHAPE: {out.shape}")

# Define the loss function
loss_function = nn.MSELoss()

# Evaluate the loss function
a = torch.rand(100)
b = torch.rand(100)
loss_value = loss_function(a, b)
print(f"Computed loss: {loss_value}")

expected_mse = np.mean((b.numpy()-a.numpy())**2)
print(f"Expected MSE: {expected_mse}")


# Since PyTorch automatically tracks the gradients, the backpropagation step can be done is a single line of code by calling the .backward() method of the loss tensor.
# Before that, you need to be sure that there are no gradients accumulated by previous operations by calling the method .zero_grad() of the network object.
x = torch.rand(128, 1).to(device)
label = torch.rand(128, 1).to(device)

# Forward pass
out = net(x)

# Compute loss
loss = loss_function(out, label)

# Backpropagation
net.zero_grad()
loss.backward()