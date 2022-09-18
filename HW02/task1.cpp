#include<iostream>
#include<cstdlib>
#include <ctime>
#include "scan.cpp"
#include <chrono>
#include <ratio>
using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration;


// Write a file task1.cpp with a main function which (in this order)
// i) Creates an array of n random float numbers between -1.0 and 1.0. n should be read as
// the first command line argument as below.
// ii) Scans the array using your scan function.
// iii) Prints out the time taken by your scan function in milliseconds 2.
// iv) Prints the first element of the output scanned array.
// v) Prints the last element of the output scanned array.
// vi) Deallocates memory when necessary.
int main(int argc, char** argv){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    int n= atoi(argv[1]);
    srand (static_cast <unsigned> (time(0)));
    float *arr= new float[n];
    float *output= new float[n];

    for(int i=0; i<n; i++){
        float random = -1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0)));
        arr[i]=random;
    }

    start = high_resolution_clock::now();
    scan(arr, output, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout <<duration_sec.count() <<endl;
    cout<<output[0]<<endl;
    cout<<output[n-1]<<endl;
    delete[] arr;
    delete[] output;
    return 0;
}