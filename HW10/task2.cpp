//2.b)

#include <random>
#include <iostream>
#include <time.h>
#include <chrono>
#include <ratio>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <random>
#include "reduce.h"


using namespace std;
using namespace chrono;

int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    

    srand(time(NULL));

    // random number generator
    random_device entropy_source;
    mt19937_64 generator(entropy_source());

    // range for random number
    const int min = -1, max = 1;

    // random number distribution
    uniform_real_distribution<float> dist(min, max);

    // create array of length n
    float *arr = new float[n];

    // fill array with random numbers
    for (int i = 0; i < n; i++)
    {
        arr[i] = dist(generator);
        // arr[i]=1;
    }
    

    MPI_Init(&argc, &argv);
    int rank;
    float global_res = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    omp_set_num_threads(t);
    float res;
    auto start = high_resolution_clock::now();    
    res = reduce(arr, 0, n);
    MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    auto end = high_resolution_clock::now();

    if (rank == 0) {
        cout << global_res <<endl;
        cout << duration_cast<duration<double, milli>>(end - start).count() << endl;
    }



}