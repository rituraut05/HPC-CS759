#include<cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

int main(int argc, char *argv[])
{   
    unsigned int n = atoi(argv[1]); // number of elements
    thrust::host_vector<float> H(n); // host vector

    // random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-1.0, 1.0);
    
    // fill host vector with random numbers
    for (unsigned int i = 0; i < n; i++){
        H[i] = dist(gen);
    }

    // copy host vector to device vector
    thrust::device_vector<float> D(n);
    thrust::copy(H.begin(), H.end(), D.begin());

    // time events
    cudaEvent_t start, stop; // start and stop events
    cudaEventCreate(&start); // create start event
    cudaEventCreate(&stop); // create stop event
    cudaEventRecord(start); // record start event

    // compute the reduce function
    auto sum = thrust::reduce(D.begin(), D.end(), (float) 0, thrust::plus<float>());

    cudaEventRecord(stop, 0); // record stop event
    cudaEventSynchronize(stop); // wait for stop event to complete
    printf("%f\n", sum); // print sum
    
    float milliseconds; // time in milliseconds
    cudaEventElapsedTime(&milliseconds, start, stop); // compute time
    printf("%f\n", milliseconds); // print time
    cudaEventDestroy(start); // destroy start event
    cudaEventDestroy(stop); // destroy stop event

    // free memory
    // H.clear();
    // D.clear();
    // H.shrink_to_fit();
    // D.shrink_to_fit();
    

    
    return 0;
}