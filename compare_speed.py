import argparse
import math

import torch
from fast_hadamard_transform_op import fast_hadamard
from hadacore_for_hip import hadacore


def parse_sizes(value):
    return [int(x) for x in value.replace(",", " ").split()]


def time_op(fn, x, scale, warmup, iters):
    for _ in range(warmup):
        fn(x, scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x, scale)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=parse_sizes, default=[64, 128, 256, 4096])
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--iters", type=int, default=3000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required")

    print(f"{'N':>6s} {'shape':>14s} {'butterfly_ms':>14s} {'hadacore_ms':>14s} {'speedup':>10s}")
    for n in args.sizes:
        shape = (n,) if args.rows == 1 else (args.rows, n)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        scale = 1.0 / math.sqrt(n)

        butterfly_ms = time_op(fast_hadamard, x, scale, args.warmup, args.iters)
        hadacore_ms = time_op(hadacore, x, scale, args.warmup, args.iters)
        speedup = butterfly_ms / hadacore_ms

        print(
            f"{n:6d} {str(tuple(shape)):>14s} "
            f"{butterfly_ms:14.6f} {hadacore_ms:14.6f} {speedup:10.3f}x"
        )


if __name__ == "__main__":
    main()
