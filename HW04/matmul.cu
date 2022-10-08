#include "matmul.cuh"

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory.
//
// A, B, and C are row major representations of nxn matrices.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float *A, const float *B, float *C, size_t n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row = tid / n;
    int col = tid % n;
    float sum = 0;
    if (row < n && col < n)
    {
        for (int i = 0; i < n; i++)
        {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// The kernel call should be followed by a call to cudaDeviceSynchronize for timing purposes.
void matmul(const float *A, const float *B, float *C, size_t n, unsigned int threads_per_block)
{
    int blocks = (n * n + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<blocks, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize();
}