#include<iostream>
#include<cstdlib>
#include <ctime>
#include "convolution.h"
#include <chrono>
#include <ratio>
using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char** argv){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    int n= atoi(argv[1]);
    srand (static_cast <unsigned> (time(0)));
    float *image= new float[n*n];
    float *output= new float[n*n];
    for(int i=0; i<n*n; i++){
        float random = -10.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(20.0)));
        image[i]=random;
    }
    int m= atoi(argv[2]);
    float *mask= new float[m*m];
    for(int i=0; i<m*m; i++){
        float random = -1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0)));
        mask[i]=random;
    }

    // n=4;m=3;
    // float image1[16]={1,3,4,8,6,5,2,4,3,4,6,8,1,4,5,2};
    // float mask1[9]={0,0,1,0,1,0,1,0,0};

    start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    cout <<duration_sec.count() <<endl;
    cout<<output[0]<<endl;
    cout<<output[n*n-1]<<endl;
    delete[] image;
    delete[] output;
    delete[] mask;
    return 0;
}