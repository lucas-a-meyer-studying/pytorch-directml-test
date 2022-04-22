# Instructions: https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-wsl

# Step 1) Install WSL2

# Step 2) Install latest NVIDIA driver (Quadro Driver after March 2022 has CUDA for DirectML enabled)

# Step 3) Install libraries: 
# sudo apt install libblas3 libomp5 liblapack3

# Step 4) Install pytorch for direct ML
# pip install pytorch-directml

import torch

tensor1 = torch.tensor([1, 2]).to("dml")
tensor2 = torch.tensor([3, 4]).to("dml")

dml_algebra = tensor1 + tensor2
print(dml_algebra)