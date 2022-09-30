/*
Q3a.)  Implement in a file calledvscale.cu, the vscale kernel function as declared and described in vscale.cuh.
This  function  should  take  in  two  arrays, a and b,  and  do  an  element-wise multiplication of the two arrays:bi=ai·bi.  
In the process, b gets overwritten.  Each thread should do at most one of the multiplication operations.
Example: a= [−5.0,2.0,1.5], b= [0.8,0.3,0.6], n= 3 The resulting b array is: b= [−4.0,0.6,0.9].
*/
#include "vscale.cuh"

// Kernel function to scale an array by a scalar
__global__ void vscale(float const *a, float *b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    b[i] = a[i] * b[i];
}