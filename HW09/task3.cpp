#include <random>
#include <iostream>
#include <time.h>
#include <chrono>
#include <ratio>
#include <cmath>
#include "mpi.h"
using namespace std;

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    srand(time(NULL));

    // random number generator
    random_device entropy_source;
    mt19937_64 generator(entropy_source());

    // range for random number
    const int min = 0, max = 1;

    // random number distribution
    uniform_real_distribution<float> dist(min, max);

    // create array of length n
    float *a = new float[n];
    float *b = new float[n];

    // fill array with random numbers
    for (int i = 0; i < n; i++)
    {
        a[i] = dist(generator);
        b[i] = dist(generator);
    }

    high_resolution_clock::time_point start; // start timer variable
    high_resolution_clock::time_point end;  // end timer variable

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // get rank of process

    double time1 = 0; // variable to store time taken
    double time2 = 0; // variable to store time taken

    if (world_rank == 0)
    {
        start = high_resolution_clock::now(); // start timer

        MPI_Send(a, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD); // send array a to process 1

        MPI_Recv(b, n, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive array b from process 1

        end = high_resolution_clock::now(); // end timer

        time1 = std::chrono::duration_cast<duration<double, std::milli>>(end - start).count(); // calculate time taken

        MPI_Send(&time1, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD); // send time taken to process 1
    }
    else if (world_rank == 1)
    {

        start = high_resolution_clock::now(); // start timer

        MPI_Recv(a, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive array a from process 0

        MPI_Send(b, n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD); // send array b to process 0

        end = high_resolution_clock::now(); // end timer
        time2 = std::chrono::duration_cast<duration<double, std::milli>>(end - start).count(); // calculate time taken

        MPI_Recv(&time1, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive time taken from process 0
        printf("%f\n", time1 + time2); // print total time taken
    }

    delete[] a;
    delete[] b;

    MPI_Finalize();
    return 0;
}
