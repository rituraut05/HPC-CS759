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
#include <iostream>
#include <random>
#include <chrono>
#include <cuda.h>

using namespace std;

int main(int argc, char *argv[]){
    int n = atoi(argv[1]); //row or column dimension of matrix
    int threads_per_block = atoi(argv[2]); //threads per block
    int size = n * n; //size of matrix
    // int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block; //blocks per grid

    //create host matrices
    float *A = new float[size]; 
    float *B = new float[size];
    float *C = new float[size];

    float *d_A, *d_B, *d_C; //device matrices

    // Allocate memory on device
    cudaMallocManaged(&d_A, size * sizeof(float));
    cudaMallocManaged(&d_B, size * sizeof(float));
    cudaMallocManaged(&d_C, size * sizeof(float));

    // Fill host matrices with random numbers
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1, 1);
    for (int i = 0; i < size; i++)
    {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    // Copy host matrices to device
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 
    cudaEventRecord(start); //start timer

    // Call matmul kernel function
    matmul(d_A, d_B, d_C, n, threads_per_block);

    cudaEventRecord(stop); //stop timer
    cudaEventSynchronize(stop); //wait for stop event to complete
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); //calculate time taken

    // Copy result from device to host
    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < size; i++){
    //     printf("%f ", A[i]);
    // }
    // printf("\n");
    // for(int i = 0; i < size; i++){
    //     printf("%f ", B[i]);
    // }
    // printf("\n");
    // for(int i = 0; i < size; i++){
    //     printf("%f ", C[i]);
    // }
    // cout<<endl;
    cout << C[size - 1] << endl;
    cout << milliseconds << endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}

