//Q1.a) Implement in a file called reduce.cu the functions reduce and reduce kernel as de-
// clared and described in reduce.cuh. Your reduce kernel should use the alteration
// from Reduction #4 (“First Add During Load” from Lecture 15). The reduce kernel
// function should be called inside the reduce function (repeatedly if needed) until the
// final sum of the entire array is obtained.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "reduce.cuh"

// implements the 'first add during global load' version (Kernel 4) for the parallel reduction
// g_idata is the array to be reduced, and is available on the device.
// g_odata is the array that the reduced results will be written to, and is available on the device.
// expects a 1D configuration.
// uses only dynamically allocated shared memory.
__global__ void reduce_kernel(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float shMemArray[]; // dynamically allocated shared memory
    unsigned int tid = threadIdx.x; // thread ID
    unsigned int bid = blockIdx.x; // block ID
    unsigned int bdim = blockDim.x; // block dimension

    // first index of element for sum reduction
    unsigned int i = bid * (bdim*2) + tid; 
    // reduced 2 elements per thread

        if (i<n){
        if (i+blockDim.x<n)
            {
                shMemArray[tid] = g_idata[i]+g_idata[i+blockDim.x];
            }
        else{
                shMemArray[tid] = g_idata[i];
        }

    }
       
    // shMemArray[tid] = g_idata[i] + g_idata[i + bdim];

    __syncthreads(); // wait for all threads to finish

    // reduction in shared memory
    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shMemArray[tid] += shMemArray[tid + s];
        }
        __syncthreads(); // wait for all threads to finish
    }

    // write result for this block to global memory
    if (tid == 0) {
        g_odata[bid] = shMemArray[0];
    }
}

// the sum of all elements in the *input array should be written to the first element of the *input array.
// calls reduce_kernel repeatedly if needed. _No part_ of the sum should be computed on host.
// *input is an array of length N in device memory.
// *output is an array of length = (number of blocks needed for the first call of the reduce_kernel) in device memory.
// configures the kernel calls using threads_per_block threads per block.
// the function should end in a call to cudaDeviceSynchronize for timing purposes
__host__ void reduce(float** input, float** output, unsigned int N, unsigned int threads_per_block) {

    while(N > 1) {
        unsigned int blocks=1; // 
        if (threads_per_block < N){
            blocks = (N + threads_per_block - 1) / (2 * threads_per_block);
            // blocks = ceil((1.0*blocks/2));
        }

        // call reduce_kernel
        reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(*input, *output, N);
        N = blocks;

        // swap input and output for next iteration
        float* temp;
        temp = *output;
        *output = *input;
        *input = temp;
    }
    
    // wait for all threads to finish
    cudaDeviceSynchronize();

}