#include "reduce.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <random>
#include <vector>

/*
2a)  Implement in a file calledreduce.cppusing the prototype specified inreduce.hthe func-tion that employs OpenMP to speed up the reduction as much as possible (i.e., use asimddirective).
*/

float reduce(const float* arr, const size_t l, const size_t r){
    float ans = 0.0;

#pragma omp parallel for simd reduction(+:ans)
    for(size_t i=l; i<r; i++){
        ans+=arr[i];
    }
        
    return ans;
}