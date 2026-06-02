import torch
from fast_hadamard_transform_op import fast_hadamard

def benchmark_hadamard(N=16384, trials=10000, dtype=torch.bfloat16):
    print(f"Benchmarking Fast Hadamard Transform: N={N}, dtype={dtype}")

    device = "cuda"
    a = torch.randn(N, device=device, dtype=dtype)

    # warm-up
    for _ in range(20):
        _ = fast_hadamard(a,1)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0

    for _ in range(trials):
        start_event.record()

        _ = fast_hadamard(a,1)

        end_event.record()
        torch.cuda.synchronize()

        total_ms += start_event.elapsed_time(end_event)

    avg_ms = total_ms / trials

    print(f"--- Results ---")
    print(f"Average Latency: {avg_ms:.6f} ms")

    return avg_ms


if __name__ == "__main__":
    benchmark_hadamard(N=4096)
    benchmark_hadamard(N=512)
