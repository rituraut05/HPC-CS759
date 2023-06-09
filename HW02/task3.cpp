#include "matmul.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstdlib>
using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    srand (static_cast <unsigned> (time(0)));
    // int n= 1000 + static_cast <int> (rand()) /( static_cast <int> (RAND_MAX/(10000)));
    int n=1024;
    double *A= new double[n*n];
    double *B= new double[n*n];
    double *C= new double[n*n];

    vector<double> vecA, vecB;

    for(int i=0; i<n*n; i++){
        double random1 = -10.0 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(20.0)));
        double random2 = -10.0 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(20.0)));
        A[i]=random1;
        B[i]=random2;
        vecA.push_back(random1);
        vecB.push_back(random2);
    }
    cout<<n<<endl;

    start = high_resolution_clock::now();
    mmul1(A,B,C,n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout<<duration_sec.count()<<endl;
    cout<<C[n*n-1]<<endl;
    delete[] C;
    C= new double[n*n]();

    start = high_resolution_clock::now();
    mmul2(A,B,C,n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout<<duration_sec.count()<<endl;
    cout<<C[n*n-1]<<endl;
    delete[] C;
    C= new double[n*n]();

    start = high_resolution_clock::now();
    mmul3(A,B,C,n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout<<duration_sec.count()<<endl;
    cout<<C[n*n-1]<<endl;
    delete[] C;
    C= new double[n*n]();

    start = high_resolution_clock::now();
    mmul4(vecA,vecB,C,n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout<<duration_sec.count()<<endl;
    cout<<C[n*n-1]<<endl;
    delete[] A;
    delete[] B;
    delete[] C;
    vecA.clear();
    vecB.clear();
    return 0;
}