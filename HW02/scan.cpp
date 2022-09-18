#include<iostream>
#include<cstdlib>
#include <ctime>

#include "scan.h"
using namespace std;
inline void scan(float* input, float* output, const int size){
    output[0]=input[0];
    for(int i=1; i<size; i++){
        output[i]=input[i]+output[i-1];
    }
}