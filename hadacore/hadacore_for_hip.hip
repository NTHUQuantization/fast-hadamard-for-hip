#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

typedef short bf16_frag __attribute__((ext_vector_type(8)));
typedef _Float16 f16_frag __attribute__((ext_vector_type(8)));
typedef float f32_frag __attribute__((ext_vector_type(8)));

#if defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(HADACORE_FORCE_GFX12_WMMA)
#define HADACORE_GFX12_WMMA 1
#elif defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__))
#define HADACORE_GFX12_WMMA 1
#else
#define HADACORE_GFX12_WMMA 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(HADACORE_FORCE_GFX12_WMMA)
#define HADACORE_HAS_F32_BF16_WMMA 1
#elif HADACORE_GFX12_WMMA && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
#define HADACORE_HAS_F32_BF16_WMMA 1
#else
#define HADACORE_HAS_F32_BF16_WMMA 0
#endif

#if defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(HADACORE_FORCE_GFX12_WMMA)
#define HADACORE_HAS_F32_F16_WMMA 1
#elif HADACORE_GFX12_WMMA && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#define HADACORE_HAS_F32_F16_WMMA 1
#else
#define HADACORE_HAS_F32_F16_WMMA 0
#endif

__device__ __forceinline__ f32_frag zero_f32_frag() {
    f32_frag frag;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = 0.0f;
    }
    return frag;
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float x) {
    uint32_t bits = __float_as_uint(x);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ _Float16 f16_bits_to_f16(uint16_t x) {
    union {
        uint16_t u;
        _Float16 h;
    } v;
    v.u = x;
    return v.h;
}

__device__ __forceinline__ uint16_t f16_to_bits(_Float16 x) {
    union {
        _Float16 h;
        uint16_t u;
    } v;
    v.h = x;
    return v.u;
}

__device__ __forceinline__ uint16_t f32_to_f16_bits(float x) {
    return f16_to_bits(static_cast<_Float16>(x));
}

__device__ __forceinline__ bf16_frag make_h16_bf16_fragment(int lane, float scale) {
    bf16_frag frag;
    int col = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = k_base + i;
        bool positive = (__builtin_popcount(static_cast<unsigned>(k & col)) & 1) == 0;
        frag[i] = static_cast<short>(f32_to_bf16_bits(positive ? scale : -scale));
    }
    return frag;
}

__device__ __forceinline__ f16_frag make_h16_f16_fragment(int lane, float scale) {
    f16_frag frag;
    int col = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = k_base + i;
        bool positive = (__builtin_popcount(static_cast<unsigned>(k & col)) & 1) == 0;
        frag[i] = static_cast<_Float16>(positive ? scale : -scale);
    }
    return frag;
}

__device__ __forceinline__ f32_frag wmma_h16_bf16_f32(bf16_frag a, bf16_frag h) {
    f32_frag c = zero_f32_frag();
#if HADACORE_HAS_F32_BF16_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, h, c);
#else
    return c;
#endif
}

__device__ __forceinline__ f32_frag wmma_h16_f16_f32(f16_frag a, f16_frag h) {
    f32_frag c = zero_f32_frag();
#if HADACORE_HAS_F32_F16_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, h, c);
#else
    return c;
#endif
}

__device__ __forceinline__ bf16_frag load_bf16_rowmajor_16x16(
    const uint16_t* base,
    int stride,
    int lane
) {
    bf16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = static_cast<short>(base[row * stride + k_base + i]);
    }
    return frag;
}

__device__ __forceinline__ f16_frag load_f16_rowmajor_16x16(
    const uint16_t* base,
    int stride,
    int lane
) {
    f16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = f16_bits_to_f16(base[row * stride + k_base + i]);
    }
    return frag;
}

__device__ __forceinline__ bf16_frag load_bf16_transposed_16x16(const uint16_t* base, int lane) {
    bf16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = static_cast<short>(base[(k_base + i) * 16 + row]);
    }
    return frag;
}

__device__ __forceinline__ f16_frag load_f16_transposed_16x16(const uint16_t* base, int lane) {
    f16_frag frag;
    int row = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag[i] = f16_bits_to_f16(base[(k_base + i) * 16 + row]);
    }
    return frag;
}

__device__ __forceinline__ void store_bf16_rowmajor_16x16(
    uint16_t* base,
    int stride,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[(row_base + i) * stride + col] = f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_f16_rowmajor_16x16(
    uint16_t* base,
    int stride,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[(row_base + i) * stride + col] = f32_to_f16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_bf16_transposed_16x16(
    uint16_t* base,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[col * 16 + row_base + i] = f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_f16_transposed_16x16(
    uint16_t* base,
    int lane,
    f32_frag frag
) {
    int col = lane & 15;
    int row_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[col * 16 + row_base + i] = f32_to_f16_bits(frag[i]);
    }
}

__device__ __forceinline__ void h256_bf16(
    const uint16_t* in,
    uint16_t* temp,
    uint16_t* out,
    int lane
) {
    bf16_frag h = make_h16_bf16_fragment(lane, 0.25f);
    f32_frag d0 = wmma_h16_bf16_f32(load_bf16_rowmajor_16x16(in, 16, lane), h);
    store_bf16_rowmajor_16x16(temp, 16, lane, d0);
    __syncthreads();
    f32_frag d1 = wmma_h16_bf16_f32(load_bf16_transposed_16x16(temp, lane), h);
    store_bf16_transposed_16x16(out, lane, d1);
    __syncthreads();
}

__device__ __forceinline__ void h256_f16(
    const uint16_t* in,
    uint16_t* temp,
    uint16_t* out,
    int lane
) {
    f16_frag h = make_h16_f16_fragment(lane, 0.25f);
    f32_frag d0 = wmma_h16_f16_f32(load_f16_rowmajor_16x16(in, 16, lane), h);
    store_f16_rowmajor_16x16(temp, 16, lane, d0);
    __syncthreads();
    f32_frag d1 = wmma_h16_f16_f32(load_f16_transposed_16x16(temp, lane), h);
    store_f16_transposed_16x16(out, lane, d1);
    __syncthreads();
}

__device__ __forceinline__ bf16_frag load_outer_bf16(const uint16_t* h256, int low_base, int lane) {
    bf16_frag frag;
    int low_off = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int chunk = k_base + i;
        frag[i] = static_cast<short>(h256[chunk * 256 + low_base + low_off]);
    }
    return frag;
}

__device__ __forceinline__ f16_frag load_outer_f16(const uint16_t* h256, int low_base, int lane) {
    f16_frag frag;
    int low_off = lane & 15;
    int k_base = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int chunk = k_base + i;
        frag[i] = f16_bits_to_f16(h256[chunk * 256 + low_base + low_off]);
    }
    return frag;
}

__device__ __forceinline__ void store_outer_bf16(
    uint16_t* out,
    int low_base,
    int lane,
    f32_frag frag
) {
    int chunk_out = lane & 15;
    int low_row = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[chunk_out * 256 + low_base + low_row + i] = f32_to_bf16_bits(frag[i]);
    }
}

__device__ __forceinline__ void store_outer_f16(
    uint16_t* out,
    int low_base,
    int lane,
    f32_frag frag
) {
    int chunk_out = lane & 15;
    int low_row = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[chunk_out * 256 + low_base + low_row + i] = f32_to_f16_bits(frag[i]);
    }
}

extern "C" __global__ void hadacore_bf16_wmma_cuda_style_4096_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
) {
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || wave >= 8) {
        return;
    }

#if HADACORE_HAS_F32_BF16_WMMA
    extern __shared__ uint32_t smem_bf16_4096_u32[];
    uint16_t* temp_all = reinterpret_cast<uint16_t*>(smem_bf16_4096_u32);
    uint16_t* h256_all = temp_all + 8 * 256;
    uint16_t* temp = temp_all + wave * 256;
    const uint16_t* in_row = in + row * 4096;
    uint16_t* out_row = out + row * 4096;

    for (int local = 0; local < 2; ++local) {
        int chunk = wave * 2 + local;
        h256_bf16(in_row + chunk * 256, temp, h256_all + chunk * 256, lane);
    }
    __syncthreads();

    bf16_frag h = make_h16_bf16_fragment(lane, 0.25f);
    for (int low_group = wave; low_group < 16; low_group += 8) {
        int low_base = low_group * 16;
        f32_frag d = wmma_h16_bf16_f32(load_outer_bf16(h256_all, low_base, lane), h);
        store_outer_bf16(out_row, low_base, lane, d);
    }
#endif
}

extern "C" __global__ void hadacore_f16_wmma_cuda_style_4096_kernel(
    const uint16_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int rows
) {
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    int row = blockIdx.x;
    if (row >= rows || wave >= 8) {
        return;
    }

#if HADACORE_HAS_F32_F16_WMMA
    extern __shared__ uint32_t smem_f16_4096_u32[];
    uint16_t* temp_all = reinterpret_cast<uint16_t*>(smem_f16_4096_u32);
    uint16_t* h256_all = temp_all + 8 * 256;
    uint16_t* temp = temp_all + wave * 256;
    const uint16_t* in_row = in + row * 4096;
    uint16_t* out_row = out + row * 4096;

    for (int local = 0; local < 2; ++local) {
        int chunk = wave * 2 + local;
        h256_f16(in_row + chunk * 256, temp, h256_all + chunk * 256, lane);
    }
    __syncthreads();

    f16_frag h = make_h16_f16_fragment(lane, 0.25f);
    for (int low_group = wave; low_group < 16; low_group += 8) {
        int low_base = low_group * 16;
        f32_frag d = wmma_h16_f16_f32(load_outer_f16(h256_all, low_base, lane), h);
        store_outer_f16(out_row, low_base, lane, d);
    }
#endif
}

extern "C" __global__ void hadacore_fallback_fp32_kernel(
    float* __restrict__ data,
    int n,
    float scale,
    int logn
) {
    extern __shared__ float tile[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float* row_data = data + row * n;

    for (int i = tid; i < n; i += blockDim.x) {
        tile[i] = row_data[i];
    }
    __syncthreads();

    for (int step = 0; step < logn; ++step) {
        int stride = 1 << step;
        int pairs = n >> 1;
        for (int p = tid; p < pairs; p += blockDim.x) {
            int q = p & (stride - 1);
            int base = ((p - q) << 1) + q;
            float a = tile[base];
            float b = tile[base + stride];
            tile[base] = a + b;
            tile[base + stride] = a - b;
        }
        __syncthreads();
    }

    for (int i = tid; i < n; i += blockDim.x) {
        row_data[i] = tile[i] * scale;
    }
}
