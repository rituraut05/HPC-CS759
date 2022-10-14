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
    int bdim = blockDim.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *shA = shMem;
    float *shB = shA + bdim * bdim;
    int index =  n * bdim * by + bdim * bx + n * ty + tx;
    int aBegin = n * bdim * by;
    int aEnd = aBegin + n - 1;
    int aStep = bdim;
    int bBegin = bdim * bx;
    int bStep = bdim * n;
    T Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
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
            Csub += shA[ty * bdim + k] * shB[k * bdim + tx];
        }

        __syncthreads();


    }

    if (index < n * n && (aBegin % n + threadIdx.x < n) && (aBegin / n + ty < n)
    && (bBegin % n + threadIdx.x < n) && (bBegin / n + threadIdx.y < n)) {
        C[index] = Csub;
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

