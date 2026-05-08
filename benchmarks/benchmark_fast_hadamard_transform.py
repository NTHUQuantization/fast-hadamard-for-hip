import torch

import torch
import torch.utils.benchmark as benchmark
import math

# from flash_attn.utils.benchmark import benchmark_forward, pytorch_profiler

def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def pytorch_profiler(
    fn,
    *inputs,
    trace_filename=None,
    backward=False,
    amp=False,
    amp_dtype=torch.float16,
    cpu=False,
    verbose=True,
    **kwinputs,
):
    """Wrap benchmark functions in Pytorch profiler to see CUDA information."""
    if backward:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
            g = torch.randn_like(out)
    for _ in range(30):  # Warm up
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        # Backward should be done outside autocast
        if backward:
            out.backward(g, retain_graph=True)
    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [
        torch.profiler.ProfilerActivity.CUDA
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)
    if verbose:
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        print(prof.key_averages().table(row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)

from fast_hadamard_transform import hadamard_transform
from fast_hadamard_transform.fast_hadamard_transform_interface import (
    hadamard_transform_12N,
    hadamard_transform_20N,
    hadamard_transform_28N,
    hadamard_transform_40N,
)

# batch_size = 16
# seqlen = 2048
# dim = 1024 * 2
# dtype = torch.float16
# device = "cuda"

# dim = 16

# torch.random.manual_seed(0)
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform, x, desc="Hadamard transform dim=2048")
# pytorch_profiler(hadamard_transform, x)
# benchmark_forward(torch.clone, x, desc="torch.clone")
# pytorch_profiler(torch.clone, x)

# dim = 2048 * 2

# torch.random.manual_seed(0)
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform, x, desc="Hadamard transform dim=4096")
# pytorch_profiler(hadamard_transform, x)
# benchmark_forward(torch.clone, x, desc="torch.clone")
# pytorch_profiler(torch.clone, x)

# dim = 4096 * 2

# torch.random.manual_seed(0)
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform, x, desc="Hadamard transform dim=8192")
# pytorch_profiler(hadamard_transform, x)
# benchmark_forward(torch.clone, x, desc="torch.clone")
# pytorch_profiler(torch.clone, x)

# dim = 8192 * 2

# torch.random.manual_seed(0)
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform, x, desc="Hadamard transform dim=16384")
# pytorch_profiler(hadamard_transform, x)
# benchmark_forward(torch.clone, x, desc="torch.clone")
# pytorch_profiler(torch.clone, x)


# dim = 16384 * 2

# torch.random.manual_seed(0)
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform, x, desc="Hadamard transform dim=32768")
# pytorch_profiler(hadamard_transform, x)
# benchmark_forward(torch.clone, x, desc="torch.clone")
# pytorch_profiler(torch.clone, x)

# dim = 12 * 512
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform_12N, x, 1.0, desc="Hadamard transform 12N")
# pytorch_profiler(hadamard_transform_12N, x, 1.0)

# dim = 20 * 1024
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform_20N, x, 1.0, desc="Hadamard transform 20N")
# pytorch_profiler(hadamard_transform_20N, x, 1.0)

# dim = 28 * 1024
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform_28N, x, 1.0, desc="Hadamard transform 28N")
# pytorch_profiler(hadamard_transform_28N, x, 1.0)

# dim = 40 * 1024
# x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
# benchmark_forward(hadamard_transform_40N, x, 1.0, desc="Hadamard transform 40N")
# pytorch_profiler(hadamard_transform_40N, x, 1.0)

batch_size = 16
seqlen = 2048
dim =  4096
dtype = torch.float16
device = "cuda"
scale = 1.0/math.sqrt(dim)
torch.random.manual_seed(0)
x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
benchmark_forward(hadamard_transform, x, desc="Hadamard transform")
pytorch_profiler(hadamard_transform, x)
benchmark_forward(torch.clone, x, desc="torch.clone")
pytorch_profiler(torch.clone, x)