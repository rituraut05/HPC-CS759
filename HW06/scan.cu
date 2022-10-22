#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include "scan.cuh"

using namespace std;

__global__ void kernel(float *g_odata,const float *g_idata, int n) {
    extern volatile __shared__ float shMem[]; // shared memory
    int tid= threadIdx.x; // thread id
    int bdim = blockDim.x; // block dimension
    int idx = blockIdx.x * blockDim.x + tid; // global thread id
    int pout = 0, pin = 1; // ping-pong buffers

    shMem[tid] = 0; // initialize shared memory
    if(idx<n) // load input into shared memory
        shMem[tid] = g_idata[idx];

    __syncthreads(); // wait for all threads to load
    
    for(int i = 1; i<bdim; i *= 2) {
        // swap double buffer indices
        pout = 1-pout; 
        pin  = 1-pout; 
        
        if(tid >= i) // check that we are not out of bounds
            shMem[pout*bdim + tid] = shMem[pin*bdim + tid] + shMem[pin*bdim + tid - i];
        else // if we are out of bounds, just copy the value
            shMem[pout*bdim + tid] = shMem[pin*bdim + tid];
        __syncthreads();
        } 
       
    if(idx<n) // write output
        g_odata[idx] = shMem[pout*bdim+tid];
}

__global__ void add_interm_output(float *input, float *output, int n) {

    int tid= threadIdx.x; // thread id
    int bdim = blockDim.x; // block dimension
    int bid = blockIdx.x; // block id
    int idx = bid * bdim + tid; // global thread id

    if (idx>n) // check that we are not out of bounds
        return;
    if (bid>0)
        input[idx] += output[bid-1];
}

__global__ void generate_interm_input_array(float *temp_out, float *output, unsigned int n, unsigned int threads_per_block) {

    unsigned int tid= threadIdx.x; // thread id
    unsigned int bdim= threads_per_block; // block dimension
    unsigned int idx= tid*bdim + bdim; // global thread id

    // 
    if(bdim<n)
        temp_out[tid] = output[idx-1];    
    else
        temp_out[tid] = output[n-1];
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block){
    unsigned int blocks = (n+threads_per_block-1)/threads_per_block;
    float * tempInput, * tempOutput;
    cudaMallocManaged((void **)&tempInput, blocks * sizeof(float)); // allocate memory for the intermediate input array
    cudaMallocManaged((void **)&tempOutput, blocks * sizeof(float)); // allocate memory for the intermediate input array

    // initialize the intermediate input and output arrays
    for(unsigned int i=0;i<blocks;i++){
        tempInput[i]=0.0;
        tempOutput[i] = 0.0;
    }

    // call the kernel to perform the scan
    int gridSize = 2*threads_per_block*sizeof(float);
    kernel<<<blocks,threads_per_block,gridSize>>>(output,input,n);
    cudaDeviceSynchronize(); // wait for the kernel to finish

    // call the kernel to create the intermediate input array
    generate_interm_input_array<<<1,blocks>>>(tempInput, output, n, threads_per_block);
    cudaDeviceSynchronize(); // wait for the kernel to finish

    // call the kernel to perform the scan on the intermediate input array
    gridSize = 2*blocks*sizeof(float);
    kernel<<<1,blocks,gridSize>>>(tempOutput, tempInput, blocks);
    cudaDeviceSynchronize(); // wait for the kernel to finish

    // call the kernel to add the intermediate output array to the output array
    add_interm_output<<<blocks,threads_per_block>>>(output, tempOutput, n); 
    cudaDeviceSynchronize(); // wait for the kernel to finish
}
