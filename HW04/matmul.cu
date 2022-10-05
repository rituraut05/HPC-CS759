/*
Q1.a)
(a)  Implement in a file called matmul.cu the matmul and matmul kernel 
functions as de-clared and described in the comment section of matmul.cuh.  
These functions shouldcompute the product of square matrices.
*/

#include "matmul.cuh"

__global__ void matmul_kernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}