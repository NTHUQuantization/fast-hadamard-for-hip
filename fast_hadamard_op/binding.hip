#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <assert.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

extern "C" __global__ void hadamard_hip_kernel(
    float* __restrict__ data,
    int N,
    float scale,
    int logN
);

// ======================================================
// PyTorch binding
// ======================================================
torch::Tensor fast_hadamard(torch::Tensor a, float s) {
    TORCH_CHECK(a.is_cuda(), "input must be CUDA/HIP tensor");

    int64_t outer = 1;
    for (int i = 0; i < a.dim() - 1; i++)
        outer *= a.size(i);

    int N = a.size(-1);

    // next power of two
    int N_padded = 1;
    while (N_padded < N) N_padded <<= 1;

    auto x = a.reshape({outer, N});

    auto padded = torch::zeros({outer, N_padded}, a.options());
    padded.narrow(1, 0, N).copy_(x);

    // compute in fp32
    auto GPU = padded.to(torch::kFloat32);
    float* ptr = GPU.data_ptr<float>();

    int logN = (int)std::log2((float)N_padded);

    dim3 grid((unsigned int)outer);
    dim3 block;
    size_t smem = N_padded * sizeof(float);

    // tuning
    if (N_padded <= 512)
        block = dim3(128);
    else
        block = dim3(256);

    hipLaunchKernelGGL(
        hadamard_hip_kernel,
        grid,
        block,
        smem,
        0,
        ptr,
        N_padded,
        s,
        logN
    );

    HIP_ASSERT(hipGetLastError());

    return GPU.reshape(a.sizes()).to(a.scalar_type());
}

// ======================================================
// module export
// ======================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_hadamard", &fast_hadamard, "Fast Hadamard Transform (HIP)");
}
