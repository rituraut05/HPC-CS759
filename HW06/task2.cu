#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include "scan.cuh"
/*
Write a test program task2.cu which does the following:
•Create and fill an array of lengthnwith randomfloatnumbers in the range [-1, 1]using managed memory, 
where n is the first command line argument as below.
•Call your scan function to fill another array with the results of the inclusive scan.
•Print the last element of the array containing the output of the inclusive scan operation.
•Print the time taken to run the full scan function in milliseconds using CUDA events.
•Compile:nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas-O3 -std c++17 -o task2
•Run  (where n is a positive integer, n≤threadsperblock*threadsperblock):./task2 n threadsperblock
•Exampled expected output:
1065.3
1.12
*/

using namespace std;

int main(int argc, char *argv[])
{   
    unsigned int n = atoi(argv[1]); //size of array
    unsigned int threads_per_block = atoi(argv[2]); //threads per block
    
    float *input = new float[n]; //input array
    float *output = new float[n]; //output array

    // random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> disFloat(-1.0, 1.0);

    //fill array with random numbers
    for (unsigned int i = 0; i < n; i++){
        input[i] = disFloat(gen);
        output[i]=disFloat(gen);
    }

    float *deviceInput, *deviceOutput;
    cudaMalloc(&deviceInput, n  * sizeof(float)); //allocate memory on device for input array
    cudaMalloc(&deviceOutput, n  * sizeof(float)); //allocate memory on device for output array

    cudaMemcpy(deviceInput, input, n*  sizeof(float), cudaMemcpyHostToDevice); //copy input array to device
    cudaMemcpy(deviceOutput, output, n * sizeof(float), cudaMemcpyHostToDevice); //copy output array to device
    
     // Create CUDA events
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); // create start event
    cudaEventCreate(&stop); // create stop event
    cudaEventRecord(start); // record start event
    scan(deviceInput,deviceOutput,n,threads_per_block); //call scan function
    cudaEventRecord(stop, 0); // record stop event
    cudaEventSynchronize(stop); // wait for stop event to complete
    cudaDeviceSynchronize(); // wait for device to finish
    cudaEventDestroy(start); // destroy start event
    cudaEventDestroy(stop); // destroy stop event
    cudaMemcpy(output, deviceOutput, n * sizeof(float), cudaMemcpyDeviceToHost); //copy output array from device to host
    cout << output[n-1] << endl; //print last element of output array
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); //calculate elapsed time
    cout << elapsedTime << endl; //print elapsed time
    cudaFree(deviceInput); //free device memory
    cudaFree(deviceOutput); //free device memory
    delete[] input; //free host memory
    delete[] output; //free host memory
    return 0;
}












