/*Write a C++ program in a file called task3.cpp which does the following:
- Launches four OpenMP threads.
- Prints out the number of threads launched, with the formatNumber of threads:  
x(followed by a newline), where x is the total number of threads.  
This should be printed only once.
- Lets each thread introduce itself, with the print formatI am thread No. i(followed bya newline), where i is the thread number.  
Each thread should do that only once.
- Computes and prints out the factorial of integers from 1 to 8,a!=b (followed by a newline),
where a is one of the 8 integers, andbis the result of a!.  
This should be done in parallelwith all 4 threads.How to go about it, 
and what the expected output looks like:
Compile:g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmpRun by submitting as batch script:./task3
Example expected output (as you can see, the order matters not):Number of threads:  4 
I am thread No.  0
I am thread No.  3
I am thread No.  1
I am thread No.  2
3!=6
5!=120
4!=24
6!=720
7!=5040
8!=40320
1!=1
2!=2
*/

#include<iostream>
#include <omp.h>
#include <stdio.h>

using namespace std;

// function to calculate factorial of given number
void factorial(int n) {
    int fact = 1; // initialize fact to 1
    for (int i = 1; i <= n; i++) 
        fact = fact * i;
    printf("%d!=%d\n", n, fact); // print factorial
}

int main(){
    int n = 8; // to calculate factorial of numbers from 1 to 8
    omp_set_num_threads(4); // set number of threads to 4
    printf("Number of threads: %d\n",omp_get_max_threads()); // print number of threads
    
    #pragma omp parallel
    {
        int thread_id  =  omp_get_thread_num(); // get thread id
        printf("I am thread No. %d \n",thread_id); // print thread id
        
    }
    #pragma omp parallel for
        for(int j=1;j<n+1;j++)
            factorial(j); // calculate factorial of numbers from 1 to 8

    return 0;
}