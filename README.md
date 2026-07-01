# Fast Hadamard Transform for ROCm (HIP)

This repository provides a fast GPU Hadamard transform implementation for AMD GPUs (ROCm / HIP) using PyTorch C++/HIP extensions.

It supports:
- float32 / float16 / bfloat16
- arbitrary batch dimensions
- automatic padding to power-of-2 length
- optional normalization scaling

---

# Installation

## 1. Clone repository
```
git clone https://github.com/david20041024/fast-hadamard-for-hip
cd fast-hadamard-for-hip
```
## 2. Environment Setup (Docker)
It is recommended to use the official vLLM ROCm development container for building the extension:
```
docker run -it --rm \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  rocm/vllm-dev:rocm7.2_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.14.0rc0
```
## 3. Build and Install
Install the HIP extension:
```
python setup.py install
```
# Validation
## 1. Orthogonality Check
- Run the test script to verify mathematical correctness. You will observe that applying the transform twice (with proper scaling) reconstructs the original input, subject to minor floating-point precision errors.
- Note: If padding is required (non-power-of-2), the sequence is truncated, which affects strict orthogonality.
```
python test.py
```
## 2. Performance Test
Measure the average GPU execution time (after warm-up) for common dimensions such as $N=2048$ and $N=16384$:
```
python speed.py
```
# How to use
- Supports float16, bfloat16, and float32 data types. The function performs high-precision computations by casting inputs to float32 internally before converting them back to their original precision
- It also supports multi-batch processing.
- To ensure the transform is its own inverse (up to precision), always use $scale = 1 / math.sqrt(N)$.
```
import math
import torch
from fast_hadamard_transform import fast_hadamard
a = torch.randn(5,32,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(fast_hadamard(a,scale))
```
# Optimized Fast Hadamard Transform for N=256 and N=4096
```
cd fast_hadamard_op
python setup.py install
```
```
import math
import torch
from fast_hadamard_transform_op import fast_hadamard
a = torch.randn(4096,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(fast_hadamard(a,scale))
```
# Hadacore
```
cd hadacore
python setup.py install
```
```
import math
import torch
from hadacore_for_hip import hadacore
a = torch.randn(4096,device="cuda",dtype=torch.float16)
scale = 1.0 / math.sqrt(a.size(-1))
print(hadacore(a,scale))
```
# compare_speed.py
Warm up both fast_hadamard and hadacore for 10,000 iterations, then benchmark their performance to verify that hadacore is faster than fast_hadamard.
