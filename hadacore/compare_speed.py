import argparse
import math

import torch
from hadacore_for_hip import hadacore


def parse_dtype(name):
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise argparse.ArgumentTypeError(f"unsupported dtype: {name}")


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
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float16)
    parser.add_argument("--baseline", choices=["none", "butterfly"], default="none")
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--iters", type=int, default=3000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required")

    baseline_fn = None
    if args.baseline == "butterfly":
        from fast_hadamard_transform_op import fast_hadamard

        baseline_fn = fast_hadamard

    print(f"dtype={args.dtype} rows={args.rows} warmup={args.warmup} iters={args.iters}")
    if baseline_fn is None:
        print(f"{'N':>6s} {'shape':>16s} {'hadacore_ms':>14s} {'Melem/s':>12s} {'GiB/s':>10s}")
    else:
        print(
            f"{'N':>6s} {'shape':>16s} {'butterfly_ms':>14s} "
            f"{'hadacore_ms':>14s} {'speedup':>10s}"
        )

    for n in args.sizes:
        shape = (n,) if args.rows == 1 else (args.rows, n)
        x = torch.randn(shape, device="cuda", dtype=args.dtype)
        scale = 1.0 / math.sqrt(float(n))

        ms = time_op(hadacore, x, scale, args.warmup, args.iters)
        elems_per_s = args.rows * n / (ms * 1e-3)
        gib_per_s = args.rows * n * x.element_size() * 2 / (ms * 1e-3) / (1024**3)

        if baseline_fn is None:
            print(f"{n:6d} {str(shape):>16s} {ms:14.6f} {elems_per_s / 1e6:12.3f} {gib_per_s:10.3f}")
        else:
            base_ms = time_op(baseline_fn, x, scale, args.warmup, args.iters)
            print(f"{n:6d} {str(shape):>16s} {base_ms:14.6f} {ms:14.6f} {base_ms / ms:10.3f}x")


if __name__ == "__main__":
    main()
