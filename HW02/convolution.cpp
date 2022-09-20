#include<iostream>
#include "convolution.h"

using namespace std;
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    int m2= (m)/2;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            float sum=0;
            for(int k=0; k<m; k++){
                for(int l=0; l<m; l++){
                    int r= i+k-m2;
                    int c= j+l-m2;
                    if((r<0 || r>=n) && (c<0 || c>=n)){
                        sum+=mask[k*m+l]*0;
                    }
                    else if(r<0 || r>=n || c<0 || c>=n){
                        sum+=mask[k*m+l];
                    }else{
                        sum+=mask[k*m+l]*image[r*n+c];
                    }
                }
            }
            output[i*n+j]=sum;
        }
    }
}