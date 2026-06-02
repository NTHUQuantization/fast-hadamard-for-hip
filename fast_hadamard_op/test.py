import math
import torch
from fast_hadamard_transform_op import fast_hadamard
a = torch.randn(4096,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(a)
print(fast_hadamard(fast_hadamard(a,scale),scale))
a = torch.randn(512,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(a)
print(fast_hadamard(fast_hadamard(a,scale),scale))
