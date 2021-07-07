import torch
a = torch.arange(1, 5)
print(a)
c = a.resize_(1, 6)
print(c)