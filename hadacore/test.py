import math
import torch
from hadacore_for_hip import hadacore

a = torch.randn(4096,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(a)
print(hadacore(hadacore(a,scale),scale))
a = torch.randn(256,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(a)
print(hadacore(hadacore(a,scale),scale))