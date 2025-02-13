import torch.nn as nn
import torch

w = torch.rand([3,5])
print(w)
a = torch.LongTensor([1,2])
emb = nn.Embedding(3,5,_weight = nn.Parameter(w))
print(emb(a))