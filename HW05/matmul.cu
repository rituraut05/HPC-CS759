#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// NOTE that each test function below calls the template matmul_kernel<TYPE>;
// The template function must meet the following requirements.
//  - Computes the matrix product C = AB using the tiled method from Lecture 11
//  - A, B, and C are row-major representations of nxn matrices in managed memory
//  - n need not be a multiple of blockDim.x
//  - Expects 2D configuration as in the slides
//  - Uses only dynamically allocated shared memory
// Function Prototype:
// __global__ void matmul_kernel(const TYPE* A, const TYPE* B, TYPE* C, unsigned int n)


template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    extern __shared__ float shMem[];
    int bdim = blockDim.x; // block dimension
    int bx = blockIdx.x; // block index x
    int by = blockIdx.y;  // block index y
    int tx = threadIdx.x; // thread index x
    int ty = threadIdx.y; // thread index y
    float *shA = shMem; // shared memory for A
    float *shB = shA + bdim * bdim; // shared memory for B
    int idx =  n * bdim * by + bdim * bx + n * ty + tx; // index of C
    int startA = n * bdim * by; // index of starting of A
    int endA = startA + n - 1; // index of ending of A
    int jumpA = bdim; // step size of A
    int startB = bdim * bx; // index of starting of B
    int jumpB = bdim * n; // step size of B
    T Cans = 0;

    for (int a = startA, b = startB; a <= endA; a += jumpA, b += jumpB) {
        int ArowInd = a / n;
        int AcolInd = a % n;
        int Aind = a + n * ty + tx;
        int Bind = b + n * ty + tx;
        int BrowInd = b / n;
        int BcolInd = b % n;

        if (Aind / n >= n || (ArowInd + ty >= n) || Aind % n < AcolInd || (AcolInd + tx >= n)) {
            shA[ty * bdim + tx] = 0;
        }
        else {
            shA[ty * bdim + tx] = A[Aind];
        }

        if (Bind / n >= n || (BrowInd + ty >= n) || Bind % n < BcolInd || (BcolInd + tx >= n))
            shB[ty * bdim + tx] = 0;
        else
            shB[ty * bdim + tx] = B[Bind];

        __syncthreads();

        for (int k = 0; k < bdim; ++k) {
            Cans += shA[ty * bdim + k] * shB[k * bdim + tx];
        }

        __syncthreads();


    }

    if (idx < n * n && (startA % n + threadIdx.x < n) && (startA / n + ty < n) && (startB % n + threadIdx.x < n) && (startB / n + threadIdx.y < n)) {
        C[idx] = Cans;
    }

}


// Matrix multiplication using the tiled method with int matrices
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    unsigned int grid_size = (n + block_dim - 1) / block_dim;
    matmul_kernel<int><<<dim3(grid_size, grid_size), dim3(block_dim, block_dim), 2 * block_dim * block_dim * sizeof(int)>>>(A, B, C, n);

    cudaDeviceSynchronize();
}

// Matrix multiplication using the tiled method with float matrices
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,unsigned int block_dim) {
    unsigned int grid_size = (n + block_dim - 1) / block_dim;
    matmul_kernel<float><<<dim3(grid_size, grid_size), dim3(block_dim, block_dim), 2 * block_dim * block_dim * sizeof(float)>>>(A, B, C, n);

    cudaDeviceSynchronize();
}

// Matrix multiplication using the tiled method with double matrices
__host__ void matmul_3(const double *A, const double *B, double *C,unsigned int n, unsigned int block_dim) {
    unsigned int grid_size = (n + block_dim - 1) / block_dim;
    matmul_kernel<double><<<dim3(grid_size, grid_size), dim3(block_dim, block_dim), 2 * block_dim * block_dim * sizeof(double)>>>(A, B, C, n);

    cudaDeviceSynchronize();
}

