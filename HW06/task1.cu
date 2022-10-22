#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include "mmul.h"
/*
Write a test file task1.cu which does the following:
•Creates three n×n matrices,A,B, and C, stored in column-major order in managed memory 
with random floatnumbers in the range [-1,  1],  where n is thefirst command line argument as below.
•Calls your mmul function n_tests times, where n_tests is the second command line argument as below.
•Prints the average time taken by a single call to mmul in milliseconds using CUDAevents.
•Compile:nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas-O3 -lcublas -std c++17 -o task1
•Run (wherenis a positive integer):./task1 n ntests
•Example expected output:
11.0
*/
using namespace std;

int main(int argc, char* argv[])
{
    unsigned int n = atoi(argv[1]); //size of matrix
    unsigned int n_tests = atoi(argv[2]); //number of tests
    float *A, *B, *C; // device matrices

    //allocate memory on device
    cudaMallocManaged(&A, sizeof(float) * n * n);
    cudaMallocManaged(&B, sizeof(float) * n * n);
    cudaMallocManaged(&C, sizeof(float) * n * n);

    //initialize random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-1.0, 1.0);

    //initialize matrices
    for (unsigned int i = 0; i < n ; i++) {
        for (unsigned int j = 0; j < n; j++) {
            A[i * n + j] = dist(gen);
            B[i * n + j] = dist(gen);
            C[i * n + j] = dist(gen);
        }
    }

    cublasHandle_t handle; //cublas handle
	cublasCreate(&handle); //create handle

    cudaEvent_t start, stop; //cuda events
    cudaEventCreate(&start); //create start event
    cudaEventCreate(&stop); //create stop event
    cudaEventRecord(start, 0); //record start event


    for (unsigned int i = 0; i < n_tests; i++)
  		mmul(handle, A, B, C, n); //call mmul

    cudaEventRecord(stop, 0); //record stop event
    cudaEventSynchronize(stop); //wait for stop event to complete
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); //calculate elapsed time
    cudaEventDestroy(start); //destroy start event
    cudaEventDestroy(stop); //destroy stop event
    float avgTime = elapsedTime/n_tests; //calculate average time
    printf("%f\n", avgTime); //print average time

    // free device memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}