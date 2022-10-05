/*
Q1.a)
(a)  Implement in a file called matmul.cu the matmul and matmul kernel 
functions as de-clared and described in the comment section of matmul.cuh.  
These functions shouldcompute the product of square matrices.
*/

#include "matmul.cuh"

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
    
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize in timing, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
    dim3 dimBlock(threads_per_block, threads_per_block);
    dim3 dimGrid((n + threads_per_block - 1) / threads_per_block, (n + threads_per_block - 1) / threads_per_block);
    matmul_kernel<<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}