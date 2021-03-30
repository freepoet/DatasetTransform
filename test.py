import pandas as pd
import numpy as np
import torch
a=np.array([[1,4234,1],[2342,2,2]])
b=np.array([[2,3,5],[2,21,234]])
c=torch.from_numpy(a)
d=c.unsqueeze(0)
print(d.shape)
a_norm = d.numpy() - np.mean(d.numpy())
b_norm = b - np.mean(b)
# print(b_norm)
r = np.sum(a_norm * b_norm) / np.sqrt(np.sum(a_norm * a_norm) * np.sum(b_norm* b_norm));
print(r)
