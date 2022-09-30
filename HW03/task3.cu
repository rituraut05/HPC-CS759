/*
Q3. b)
Write a file task3.cu which does the following:
•Creates two arrays of length n filled by random numbers where n is read from the first command line argument.
The range of values for array a is [-10.0, 10.0], whereas the range of values for array b is [0.0, 1.0].
•Calls  your vscale kernel  with  a  1D  execution  configuration  that  uses  512  threads  perblock.
•Prints the amount of time taken to execute the kernel in milliseconds using CUDA events2.
•Prints the first element of the resulting array.
•Prints the last element of the resulting array.
*/

#include "vscale.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);

    // Defined arrays h_a, h_b on host
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *d_a, *d_b; // Pointer to arrays d_a, d_b on device

    // Allocated memory for arrays d_a, d_b on device
    cudaMallocManaged(&d_a, n * sizeof(float));
    cudaMallocManaged(&d_b, n * sizeof(float));

    // Initialize random number generator
    random_device entropy_source;
    mt19937_64 generator(entropy_source());

    // Set the ranges of the random numbers to be generated according to the given ranges
    const float minA = -10.0, maxA = 10.0;
    const float minB = 0.0, maxB = 1.0;

    // Define the distribution
    uniform_real_distribution<float> distA(minA, maxA);
    uniform_real_distribution<float> distB(minB, maxB);

    // Generate random numbers and store them in the host arrays
    for (int i = 0; i < n; i++)
    {
        h_a[i] = distA(generator);
        h_b[i] = distB(generator);
    }

    // Copy the host arrays to the device arrays
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    int numThreads = 512; // Number of threads per block
    // int numThreads = 16;
    int numBlocks = (n + numThreads - 1) / numThreads; // Number of blocks per grid

    // Create CUDA events to measure the time taken to execute the kernel
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call the kernel
    vscale<<<numBlocks, numThreads>>>(d_a, d_b);

    cudaDeviceSynchronize();    // Wait for the kernel to finish
    cudaEventRecord(stop);      // Record the time when the kernel finishes
    cudaEventSynchronize(stop); // Wait for the stop event to finish

    float ms;
    cudaEventElapsedTime(&ms, start, stop); // Calculate the time taken to execute the kernel

    // Copy the result from the device to the host
    cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Prints the amount of time taken to execute the kernel in ms,
    // and the first and last elements of the resulting array
    printf("%f\n", ms);
    printf("%f\n", h_b[0]);
    printf("%f\n", h_b[n - 1]);

    // Free the memory allocated on the host
    delete[] h_a;
    delete[] h_b;
    // Free the memory allocated on the device
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}