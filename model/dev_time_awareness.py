import torch

row = torch.tensor([0,5,3]) 
col = torch.tensor([5,6,1])
c = row < col
print(c)