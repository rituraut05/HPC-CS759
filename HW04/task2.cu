#include "stencil.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

using namespace std;
int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);
    int R = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1, 1);

    float *image, *output, *mask;
    int mask_size = 2 * R + 1;

    cudaMallocManaged((void **)&image, n * sizeof(float));
    cudaMallocManaged((void **)&output, n * sizeof(float));
    cudaMallocManaged((void **)&mask, mask_size * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        image[i] = static_cast<float>(dis(gen));
    }

    for (int i = 0; i < mask_size; ++i)
    {
        mask[i] = static_cast<float>(dis(gen));
    }

    cudaEvent_t start;       // start timer
    cudaEvent_t stop;        // stop timer
    cudaEventCreate(&start); // create start event
    cudaEventCreate(&stop);  // create stop event

    cudaEventRecord(start); // start timer

    stencil(image, mask, output, n, R, threads_per_block);

    cudaEventRecord(stop);      // stop timer
    cudaEventSynchronize(stop); // wait for stop event to complete
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); // calculate elapsed time
    printf("%f\n", output[n - 1]);
    printf("%f\n", milliseconds); // print elapsed time

    cudaFree(image);
    cudaFree(output);
    cudaFree(mask);
}
