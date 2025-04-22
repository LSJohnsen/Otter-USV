import torch


print(torch.version.cuda)         # Shows CUDA version PyTorch is built with
print(torch.cuda.is_available())  # True = GPU available and CUDA works
print(torch.cuda.get_device_name()) 

print(torch.backends.cudnn.version())
device = torch.device("cuda")
print(f"Using device: {device}")    