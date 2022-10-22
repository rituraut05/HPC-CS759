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
    
    float *deviceInput, *deviceOutput; //device arrays

    //allocate memory on device
    cudaMallocManaged((void **)&deviceInput, n * sizeof(float));
    cudaMallocManaged((void **)&deviceOutput, n * sizeof(float));

    
    //generating random number.
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);
    for (unsigned int i = 0; i < n; i++){
        deviceInput[i] = dis(gen);
        deviceOutput[i]=dis(gen);
    }    

    cudaEvent_t start, stop; //start and stop events
    cudaEventCreate(&start); //create start event
    cudaEventCreate(&stop); //create stop event
    cudaEventRecord(start); //record start event
    scan(deviceInput,deviceOutput,n,threads_per_block); //call scan function
    cudaEventRecord(stop, 0); //record stop event
    cudaEventSynchronize(stop); //wait for stop event to complete
    cout<<deviceOutput[n-1]<<endl; //print last element of output array
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); //calculate elapsed time
    cout<<elapsedTime<<endl; //print elapsed time
    cudaEventDestroy(start); //destroy start event
    cudaEventDestroy(stop); //destroy stop event
    cudaFree(deviceInput); //free device memory
    cudaFree(deviceOutput); //free device memory
    return 0;
}












