print("Testing pytorch")

import torch

print("Cuda", torch.version.cuda)
print("Torch", torch.version.__version__)
print("CXX11abi", torch._C._GLIBCXX_USE_CXX11_ABI)

import sys
print(sys.version)

print(torch.cuda.get_arch_list())

print("cuda available", torch.cuda.is_available())

print(f'PyTorch cuDNN Version: {torch.backends.cudnn.version()}')

if torch.cuda.is_available():
    print("Pytorch okay")

import torch
import torch.nn as nn
 
# Check if CUDA is available
device = torch.device("cuda")
 
# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])
# Move the tensor to the GPU
x = x.to(device)
print(f"Tensor x is on: {x.device}")
 
# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3, 1)
 
    def forward(self, x):
        return self.fc(x)
 
model = SimpleModel()
# Move the model to the GPU
model = model.to(device)
print(f"Can upload to GPU: {next(model.parameters()).device}")


print("Testing causal_conv1d")

from causal_conv1d import causal_conv1d_fn

print("Imported causal_conv1d")

batch, dim, seq, width = 10, 5, 17, 4
x = torch.zeros((batch, dim, seq)).to('cuda')
weight = torch.zeros((dim, width)).to('cuda')
bias = torch.zeros((dim, )).to('cuda')

causal_conv1d_fn(x, weight, bias, None)

print("Causal okay")
print("Testing mamba")

from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape

print("Mamba okay")
