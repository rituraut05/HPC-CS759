// //Q1.b) Write a test program task1.cu which will complete the following (some memory man-
// // agement steps are omitted for clarity, but you should implement them in your code):
// // •Create and fill an array of length N with random numbers in the range [-1,1] on the
// // host, where N is the first command line argument as below.
// // •Copy this host array to device as the input array where the reduction will be
// // performed on.
// // •Create another output array on the device that has its length equal to the number
// // of blocks required for the first call to the kernel function reduce kernel.
// // •Call your reduce function to sum all the elements in the input array, with the
// // threads per block read from the second command line argument as below.
// // •Print the resulting sum.
// // •Print the time taken to run the reduce function in milliseconds.
// // •Compile: nvcc task1.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas
// // -O3 -std c++17 -o task1
// // •Run (where N ≤ 230 and threads per block are positive integers, and N is not
// // necessarily a power of 2):
// // ./task1 N threads per block


#include <iostream>
#include <random>
#include "reduce.cuh"

using namespace std;

int main(int argc, char* argv[]) {
    unsigned int n = atoi(argv[1]); // number of elements in the array
    unsigned int threadsPerBlock = atoi(argv[2]); // number of threads per block
    unsigned int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // number of blocks per grid
        
        float *h_input = new float[n]; // host input array
        float *d_input, *d_output; // device input and output arrays

        // allocate memory on the device
        cudaMalloc((void**)&d_input, n * sizeof(float));
        cudaMalloc((void**)&d_output, blocksPerGrid * sizeof(float));

        // fill the host input array with random numbers
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1, 1);
        for (int i = 0; i < n; i++) {
            h_input[i] = dis(gen);
            // h_input[i] = 1;
        }

        // copy the host input array to the device
        cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, blocksPerGrid * sizeof(float));
        
        cudaEvent_t start, stop; // events for timing
        cudaEventCreate(&start); // create start event
        cudaEventCreate(&stop); // create stop event
        cudaEventRecord(start, 0); // record start event
        reduce(&d_input, &d_output, n, threadsPerBlock); // call reduce function
        cudaEventRecord(stop, 0); // record stop event
        cudaEventSynchronize(stop); // wait for stop event to complete
        float elapsedTime; // time in milliseconds
        cudaEventElapsedTime(&elapsedTime, start, stop); // compute elapsed time
        cudaEventDestroy(start); // destroy start event
        cudaEventDestroy(stop); // destroy stop event
        
        float res;
        cudaMemcpy(&res, d_input, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost); // copy result from device to host
        printf("%f\n", res); // print result
        printf("%f\n", elapsedTime); // print time taken in milliseconds

        // free memory on the device
        cudaFree(d_input);
        cudaFree(d_output);

        // free memory on the host
        delete[] h_input;
        return 0;

  return 0;
}