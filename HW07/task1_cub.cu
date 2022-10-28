#define CUB_STDERR
#include <stdio.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
#include <cuda.h>

using namespace std;

cub::CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

int main(int argc, char *argv[])
{
    unsigned int n = atoi(argv[1]); // Number of items to reduce

    // random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-1.0, 1.0);

    float *hostInput = new float[n]; // input array

    for (unsigned int i = 0; i < n; i++) {
        hostInput[i] = dist(gen); // random number between -1 and 1
    }

    float *deviceInput = NULL; // device input array
    g_allocator.DeviceAllocate((void **)&deviceInput, sizeof(float) * n); // allocate device memory
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * n, cudaMemcpyHostToDevice); // copy input array to device

    float *deviceOutput = NULL; // device output array
    g_allocator.DeviceAllocate((void **)&deviceOutput, sizeof(float) * 1); // allocate device memory

    // Declare temporary storage
    void *deviceTemp = NULL; 
    size_t deviceTempSize = 0; 

    // Allocate temporary storage
    cub::DeviceReduce::Sum(deviceTemp, deviceTempSize, deviceInput, deviceOutput, n);
    g_allocator.DeviceAllocate(&deviceTemp, deviceTempSize);

    // Cuda event to measure time
    cudaEvent_t start, stop; // start and stop events
    cudaEventCreate(&start); // create start event
    cudaEventCreate(&stop); // create stop event
    cudaEventRecord(start); // record start event

    // Run reduce
    cub::DeviceReduce::Sum(deviceTemp, deviceTempSize, deviceInput, deviceOutput, n);

    cudaEventRecord(stop, 0); // record stop event
    cudaEventSynchronize(stop); // wait for stop event to complete
    float gpu_sum;
    cudaMemcpy(&gpu_sum, deviceOutput, sizeof(float) * 1, cudaMemcpyDeviceToHost); // copy output array to host
    printf("%f\n", gpu_sum); // print sum

    // Print time taken
    float milliseconds; // time taken
    cudaEventElapsedTime(&milliseconds, start, stop); // calculate time
    printf("%f\n", milliseconds); // print time
    cudaEventDestroy(start); // destroy start event
    cudaEventDestroy(stop); // destroy stop event

    // Free device memory
    if (deviceInput)
        g_allocator.DeviceFree(deviceInput); 
    if (deviceOutput)
        g_allocator.DeviceFree(deviceOutput);
    if (deviceTemp)
        g_allocator.DeviceFree(deviceTemp);

    return 0;
}