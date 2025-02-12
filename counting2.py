import torch
import torchvision
print(torch.cuda.is_available())  # 如果返回 True，说明 CUDA 可以被正确使用
print(torch.__version__)
print(torchvision.__version__)
