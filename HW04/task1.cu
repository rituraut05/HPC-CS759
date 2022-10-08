/*
Write a program task1.cu which will complete the following (some
memory manage-ment steps are omitted for clarity, but you should
implement them in your code for it to work properly):Create matrices
(as 1D row major arrays)A and B of sizen×n on the host.
1. Create matrices (as 1D row major arrays)A and B of size n×n on the host.
2. Fill these matrices with random numbers in the range [-1, 1]
3. Prepare arrays that are allocated as device memory
    (they will be passed to your matmul function.)
4. Call your matmul function.
5. Print the last element of the resulting matrix.
6. Print the time taken to execute your matmul function
    in milliseconds using CUDA events.
7.Compile: nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1
8. Run (where n and threads per block are positive integers): ./task1 n threads per block – Note n is not necessarily a power of 2.
– Use Slurm to run your job on Euler
9. Example expected output: -16.35
1.23
*/
#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

using namespace std;

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);               // row or column dimension of matrix
    int threadsPerBlock = atoi(argv[2]); // threads per block

    float *dA, *dB, *dC; // device matrices

    // random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1, 1);

    // Allocate memory on device
    cudaMallocManaged(&dA, n * n * sizeof(float));
    cudaMallocManaged(&dB, n * n * sizeof(float));
    cudaMallocManaged(&dC, n * n * sizeof(float));

    // Fill matrices with random numbers
    for (int i = 0; i < n * n; i++)
    {
        dA[i] = dis(gen);
        dB[i] = dis(gen);
        dC[i] = 0;
    }

    cudaEvent_t start;       // start time
    cudaEvent_t stop;        // end time
    cudaEventCreate(&start); // create start time event
    cudaEventCreate(&stop);  // create end time event
    cudaEventRecord(start);  // record start time

    matmul(dA, dB, dC, n, threadsPerBlock); // call matmul function

    cudaEventRecord(stop);      // record end time
    cudaEventSynchronize(stop); // wait for stop event to complete

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); // calculate time taken

    printf("%f\n", dC[n * n - 1]); // print last element of resulting matrix
    printf("%f\n", milliseconds);  // print time taken

    // free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
