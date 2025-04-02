import torch

torch.zeros(1000, 1000, device="cuda")  # This might trigger cudaMalloc
print(torch.cuda.memory_summary(device="cuda", abbreviated=False))

