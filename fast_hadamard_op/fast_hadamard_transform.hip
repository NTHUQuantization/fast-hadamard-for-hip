#include <hip/hip_runtime.h>

extern "C" __global__
void hadamard_hip_kernel(
    float* __restrict__ data,
    int N,
    float scale,
    int logN
) {
    // Shared memory size should be N * sizeof(float)
    extern __shared__ float sdata[];

    const int tid  = threadIdx.x;
    const int lane = tid & 31; // Based on 32-thread granularity
    const int row  = blockIdx.x;

    // Pointer to the start of the current row
    float* base = data + row * N;

    // ==================================================
    // CASE 1: SMALL N (N <= 1024)
    // Entire transform performed in LDS
    // ==================================================
    if (N <= 1024) {
        // Load global memory to LDS
        for (int i = tid; i < N; i += blockDim.x) {
            sdata[i] = base[i];
        }
        __syncthreads();

        // Standard iterative butterfly in LDS
        for (int step = 0; step < logN; step++) {
            int stride = 1 << step;
            for (int i = tid; i < N; i += blockDim.x) {
                if ((i & stride) == 0) {
                    int pair = i | stride;
                    float a = sdata[i];
                    float b = sdata[pair];
                    sdata[i]    = a + b;
                    sdata[pair] = a - b;
                }
            }
            __syncthreads();
        }

        // Write back with scaling
        for (int i = tid; i < N; i += blockDim.x) {
            base[i] = sdata[i] * scale;
        }
        return;
    }

    // ==================================================
    // CASE 2: LARGE N (e.g., 4096)
    // Optimized with Registers and Shuffles
    // ==================================================
    // Assuming blockDim.x = 256 and N = 4096, ELEMS = 16
     const int ELEMS = 16;   // 4096 / 256
    float r[ELEMS];

    // -----------------------------
    // load
    // -----------------------------
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) {
        int idx = tid + i * blockDim.x;
        r[i] = base[idx];
    }

    // ==================================================
    // local butterfly (0~3)
    // ==================================================
    for (int step = 0; step < 4; step++) {
        int stride = 1 << step;

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            if ((i & stride) == 0) {
                float a = r[i];
                float b = r[i + stride];

                r[i]          = a + b;
                r[i + stride] = a - b;
            }
        }
    }

    // ==================================================
    // warp shuffle stage (4~8)
    // FIXED: correct HIP API (__shfl_xor_sync)
    // ==================================================
    for (int step = 4; step < 8; step++) {
        int mask = 1 << (step - 4);

        #pragma unroll
        for (int i = 0; i < ELEMS; i++) {
            float a = r[i];

            float b = __shfl_xor_sync(0xffffffffULL, a, mask, 32);

            if ((lane & mask) == 0)
                r[i] = a + b;
            else
                r[i] = b - a;
        }
    }

    // ==================================================
    // store to LDS
    // ==================================================
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) {
        int idx = tid + i * blockDim.x;
        sdata[idx] = r[i];
    }

    __syncthreads();

    // ==================================================
    // LDS stage (8~logN)
    // ==================================================
    for (int step = 8; step < logN; step++) {
        int stride = 1 << step;

        for (int i = tid; i < N / 2; i += blockDim.x) {
            int idx  = (i / stride) * (2 * stride) + (i % stride);
            int pair = idx + stride;

            float a = sdata[idx];
            float b = sdata[pair];

            sdata[idx]  = a + b;
            sdata[pair] = a - b;
        }

        __syncthreads();
    }

    // ==================================================
    // store back
    // ==================================================
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) {
        int idx = tid + i * blockDim.x;
        base[idx] = sdata[idx] * scale;
    }
}
