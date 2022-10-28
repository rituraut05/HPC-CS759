/*
 Write a test programtask2.cu which does the following:
 Create and fill with randomintnumbers in the range [0, 500] athrust::hostvector of length n wheren is the 
 first command line argument as below.
 Use the built-in function in Thrustto copy thethrust::hostvectorinto athrust::devicevectoras the input of yourcountfunction.
 Allocate two otherthrust::devicevectors,valuesandcounts, then call yourcountfunction to fill these two arrays with 
 the results of this counting operation.
 Print the last element ofvaluesarray.
 Print the last element ofcountsarray.
 Print the time taken to run the count function in milliseconds using CUDA events.
 Compile:nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3-std c++17 -o task2
 Run by submitting asbatchscript (where n is a positive integer):./task2 n
 Example expected output:370230.13
*/

#include<cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<thrust/sort.h>
#include "count.cuh"

using namespace std;

int main(int argc, char *argv[])
{   
    unsigned int n = atoi(argv[1]); //size of vector
    
    thrust::host_vector<int> hostInput(n);

    // random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, 500);
    
    //fill vector with random numbers
    for (unsigned int i = 0; i < n; i++){
        hostInput[i] =dis(gen);
    }

    // device vectors for input and output
    thrust::device_vector<int> deviceInput(n); 
    thrust::device_vector<int> values(n);
    thrust::device_vector<int> counts(n);
    

    // copy data to the device
    thrust::copy(hostInput.begin(), hostInput.end(), deviceInput.begin());
    
    // cuda event variables
    cudaEvent_t start, stop; // cuda start and stop events
    cudaEventCreate(&start); // create start event
    cudaEventCreate(&stop); // create stop event
    cudaEventRecord(start); // record start event

    // call count function
    count(deviceInput,values,counts);

    cudaEventRecord(stop, 0); // record stop event
    cudaEventSynchronize(stop); // wait for stop event to complete

    cout<<values.back()<<endl;
    cout<<counts.back()<<endl;
    // printf("%d\n", values.back());
    // printf("%d\n", counts.back());
    
    float milliseconds; // time taken in milliseconds
    cudaEventElapsedTime(&milliseconds, start, stop); // calculate time taken
    printf("%f\n", milliseconds); // print time taken
    cudaEventDestroy(start); // destroy start event
    cudaEventDestroy(stop); // destroy stop event

    return 0;
}