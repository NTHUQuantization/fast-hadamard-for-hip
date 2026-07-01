#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <assert.h>
#include <cmath>
#include <stdint.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

extern "C" __global__ void hadacore_bf16_wmma_cuda_style_4096_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
);

extern "C" __global__ void hadacore_f16_wmma_cuda_style_4096_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
);

extern "C" __global__ void hadacore_fallback_fp32_kernel(
    float* __restrict__ data,
    int n,
    float scale,
    int logn
);

constexpr bool is_power_of_two(uint32_t x) {
    return x && !(x & (x - 1));
}

static int log2_exact(int n) {
    int logn = 0;
    while ((1 << logn) < n) {
        ++logn;
    }
    return logn;
}

static dim3 fallback_block(int n) {
    return n <= 512 ? dim3(128) : dim3(256);
}

static void launch_fp32_fallback(torch::Tensor work, int n, hipStream_t stream) {
    size_t smem = static_cast<size_t>(n) * sizeof(float);
    TORCH_CHECK(smem <= 64 * 1024, "fallback supports sizes up to 16384");

    hipLaunchKernelGGL(
        hadacore_fallback_fp32_kernel,
        dim3(static_cast<unsigned int>(work.size(0))),
        fallback_block(n),
        smem,
        stream,
        work.data_ptr<float>(),
        n,
        1.0f / std::sqrt(static_cast<float>(n)),
        log2_exact(n)
    );
    HIP_ASSERT(hipGetLastError());
}

static torch::Tensor hadacore_normalized(at::Tensor& in) {
    TORCH_CHECK(in.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(in.is_cuda(), "input must be a CUDA/HIP tensor");

    auto dtype = in.scalar_type();
    TORCH_CHECK(
        dtype == torch::ScalarType::Float ||
        dtype == torch::ScalarType::Half ||
        dtype == torch::ScalarType::BFloat16,
        "only fp32, fp16 and bf16 are supported"
    );

    int n = static_cast<int>(in.size(-1));
    TORCH_CHECK(is_power_of_two(static_cast<uint32_t>(n)), "Hadamard size must be power of two");
    TORCH_CHECK(n <= (1 << 14), "Hadamard size must be <= 16384");

    auto result_shape = in.sizes();
    torch::Tensor x = in.reshape({-1, n});
    if (x.stride(-1) != 1) {
        x = x.contiguous();
    }

    at::hip::HIPGuard device_guard{static_cast<char>(x.get_device())};
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    if (n == 4096 && dtype == torch::ScalarType::Half) {
        torch::Tensor out = torch::empty_like(x);
        hipLaunchKernelGGL(
            hadacore_f16_wmma_cuda_style_4096_kernel,
            dim3(static_cast<unsigned int>(x.size(0))),
            dim3(8 * 32),
            (8 * 256 + 16 * 256) * sizeof(uint16_t),
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()),
            static_cast<int>(x.size(0))
        );
        HIP_ASSERT(hipGetLastError());
        return out.reshape(result_shape);
    }

    if (n == 4096 && dtype == torch::ScalarType::BFloat16) {
        torch::Tensor out = torch::empty_like(x);
        hipLaunchKernelGGL(
            hadacore_bf16_wmma_cuda_style_4096_kernel,
            dim3(static_cast<unsigned int>(x.size(0))),
            dim3(8 * 32),
            (8 * 256 + 16 * 256) * sizeof(uint16_t),
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()),
            static_cast<int>(x.size(0))
        );
        HIP_ASSERT(hipGetLastError());
        return out.reshape(result_shape);
    }

    torch::Tensor work = dtype == torch::ScalarType::Float ? x.clone() : x.to(torch::kFloat32);
    launch_fp32_fallback(work, n, stream);
    torch::Tensor out = dtype == torch::ScalarType::Float ? work : work.to(dtype);
    return out.reshape(result_shape);
}

torch::Tensor hadacore(at::Tensor& in, double scale) {
    torch::Tensor out = hadacore_normalized(in);
    double default_scale = 1.0 / std::sqrt(static_cast<double>(in.size(-1)));
    double ratio = scale / default_scale;
    if (std::abs(ratio - 1.0) < 1e-12) {
        return out;
    }
    return out * ratio;
}

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hadacore", &hadacore, py::arg("x"), py::arg("scale"));
}
