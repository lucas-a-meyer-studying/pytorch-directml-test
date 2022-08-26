import torch

tensor1 = torch.tensor([1, 2]).to("cuda")
tensor2 = torch.tensor([3, 4]).to("cuda")

cuda_algebra = tensor1 + tensor2
print(cuda_algebra)