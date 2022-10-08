/*
 Q.2.a) Implement in a file called stencil.cu the stencil and stencil kernel
 functions as declared and described in the comment section of stencil.cuh.
 These functions should produce the 1D convolution of image and mask as the
 following: output[i] =RXj=−Rimage[i+j]∗mask[j+R]i= 0,···,n−1.
 Assume that image[i] = 1 when i <0 or i >n−1.
 Pay close attention to what data you are asked to store and compute in
 shared memory.
*/

#include "stencil.cuh"
#include <cuda.h>

__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n, unsigned int R_)
{

    extern __shared__ float shMemArray[]; // shared memory array

    int R = (int)R_;                         // radius of mask
    int tid = threadIdx.x;                   // thread id
    int idx = tid + blockIdx.x * blockDim.x; // id of the image pixel the thread

    if (idx >= n)
    {
        return;
    }

    // setting the starting point of the sh_mask array in shared memory
    float *sh_mask = shMemArray;

    // Initialize the sh_mask array
    for (int i = -R; i <= R; i++)
    {
        sh_mask[i + R] = mask[i + R];
    }

    // setting the starting point of the sh_output array
    //  with reference to sh_mask array
    float *sh_output = 2 * R + 1 + sh_mask;
    sh_output[tid] = 0; // initialize the sh_output array element

    // setting the starting point of the sh_image array
    float *sh_image = sh_output + blockDim.x;

    // Initialize the sh_image array
    for (int i = -R; i <= R; i++)
    {
        if (idx + i < 0 || idx + i > n - 1)
        { // if the index is out of bounds, set the value to 1
            sh_image[tid + R + i] = 1;
        }
        else
        {
            sh_image[tid + R + i] = image[idx + i];
        }
    }

    // synchronize all threads
    __syncthreads();

    // compute the convolution
    for (int i = -R; i <= R; i++)
    {
        sh_output[tid] += sh_mask[i + R] * sh_image[tid + R + i];
    }

    // copy the result to the output array
    output[idx] = sh_output[tid];
}

__host__ void stencil(const float *image,
                      const float *mask,
                      float *output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block)
{

    int shared_size = (1 + 4 * R + 2 * threads_per_block)* sizeof(float);                                   // size of shared memory array
    int numBlock = (n - 1 + threads_per_block) / threads_per_block;                          // number of blocks
    stencil_kernel<<<numBlock, threads_per_block, shared_size>>>(image, mask, output, n, R); // call the kernel function
    cudaDeviceSynchronize();
}