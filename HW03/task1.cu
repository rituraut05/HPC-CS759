#include <iostream>
using namespace std;

/*
Q1. Write a C++ program using CUDA in a file calledtask1.cuwhich computes the factorial of integersfrom 1 to 8, by launching a GPU kernel with 1 block and 8 threads.  Inside the kernel, each threadshould usestd::printfto write outa!=b(followed by a newline), whereais one of the 8 integers,andbis the result ofa!.  (Follow your kernel call with a call tocudaDeviceSynchronize()so thatthe host waits for the kernel to finish printing before returning frommain.)•Compile:nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -otask1•Run (on Euler, use Slurm sbatch!):./task1•Expected output (showing only 4 out of the 8 lines expected; lines could be out of order):1!=12!=23!=64!=24
*/

// Kernel function to find the factorial of the given number
__global__ void factorial()
{
    int tid = threadIdx.x;
    int fact = 1;
    for (int i = 1; i <= tid + 1; i++)
    {
        fact *= i;
    }
    printf("%d!=%d\n", tid + 1, fact);
}

int main()
{
    int n = 8;

    // Launch kernel on 1 block with 8 threads
    factorial<<<1, n>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    return 0;
}