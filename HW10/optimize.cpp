
/*
Write five optimization functions in optimize.cpp that each (either represents the baseline,or) 
uses a different technique to capitalize on ILP as follows:
optimize1 will be the same as reduce4 function in slide 20.
optimize2 will be the same as unroll2areduce function in slide 31.
optimize3 will be the same as unroll2aareduce function in slide 33.
optimize4 will be the same as unroll2areduce function in slide 36.
optimize5 will be similar to reduce4, but with K= 3 and L= 3, where K and L are the parameters defined in slide 39.
*/

#include "optimize.h"
#include <cstddef>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>

using namespace std;

data_t *get_vec_start(vec *v)
{
    return v->data;
}

void optimize1(vec *v, data_t *dest)
{
    size_t length = v->len;
    data_t *d = get_vec_start(v);
    data_t temp = IDENT;
    for(size_t i = 0; i < length; i++)
        temp = temp OP d[i];
    *dest= temp;
}

void optimize2(vec *v, data_t *dest)
{
    size_t length = v->len;
    size_t limit = length -1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    size_t i;
    // reduce 2 elements at a time 
    for(i = 0; i < limit; i += 2) 
    {
        x = (x OP d[i]) OP d[i + 1];
    }
    // Finish any remaining elements
    for(; i < length; i++) 
    {
        x = x OP d[i];
    }
    *dest= x;
}

void optimize3(vec *v, data_t *dest)
{
    size_t length = v->len;
    size_t limit = length -1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    size_t i;
    // reduce 2 elements at a time 
    for(i = 0; i < limit; i += 2) 
    {
        x = x OP(d[i] OP d[i + 1]);
    }
    // Finish any remaining elements
    for(; i < length; i++) 
    {
        x = x OP d[i];
    }
    *dest= x;
}

void optimize4(vec *v, data_t *dest)
{
    size_t length = v->len;
    size_t limit = length -1;
    data_t *d = get_vec_start(v);
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    size_t i;
    // reduce 2 elements at a time 
    for(i = 0; i < limit; i += 2) 
    {
        x0 = x0 OP d[i];
        x1 = x1 OP d[i + 1];
    }
    // Finish any remaining elements
    for(; i < length; i++) 
    {
        x0 = x0 OP d[i];
    }
    *dest= x0 OP x1;
}

void optimize5(vec *v, data_t *dest) 
{
    size_t length = v->len;
    size_t limit = length -1;
    data_t *d = get_vec_start(v);
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    data_t x2 = IDENT;
    size_t i;
    // reduce 2 elements at a time 
    for(i = 0; i < limit; i += 3) 
    {
        x0 = x0 OP d[i];
        x1 = x1 OP d[i + 1];
        x2 = x2 OP d[i + 2];
    }
    // Finish any remaining elements
    for(; i < length; i++) 
    {
        x0 = x0 OP d[i];
    }
    *dest= x0 OP x1 OP x2;
}