#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    // Write the float32 results into output; this kernel returns no value.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float local = input[i];
        output[i] = local / (1 + expf(-local));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
