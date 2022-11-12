/**
 Write a programtask1.cppthat will accomplish the following:
 •Create and fill withfloattype random numbers an arrayarrof lengthnwherenisthe first command line argument, see below.
 The range for these random floating pointnumbers is [0,n].
 •Sortarr(std::sortwill do).
 •Create and fill an arraycentersof lengtht, wheretis the number of threads and it’sthe second command line argument, see below.
 Assuming thatnis always a multipleof2 * t(your code does not need to account for the cases wherenis not a multipleof2 * t),
 the entries in the arraycentersshould be (as floating point numbers): n2t,3n2t,...,(2t−1)n2t.
 •Create and initialize with zeros an arraydistsof lengtht1, wheretis the number ofthreads and it’s the second command line argument, see below.
 •Call theclusterfunction and save the output distances to thedistsarray.
 •Calculate the maximum distance in thedistsarray.
 •Print the maximum distance.•Print the partition ID (the thread number) that has the maximum distance.
**/

#include <random>
#include <iostream>
#include <time.h>
#include <chrono>
#include <ratio>
#include <cmath>
#include "cluster.h"
#include <algorithm>
#include <cassert>
using namespace std;
using namespace chrono;

int main(int argc, char *argv[])
{
    size_t n = atoi(argv[1]);
    size_t t = atoi(argv[2]);

    srand(time(NULL));

    // random number generator
    random_device entropy_source;
    mt19937_64 generator(entropy_source());

    // range for random number
    const int min = 0, max = n;

    // random number distribution
    uniform_real_distribution<float> dist(min, max);

    // create array of length n
    float *array = new float[n];

    // fill array with random numbers
    for (size_t i = 0; i < n; i++)
    {
        array[i] = dist(generator);
    }

    // array for centers and distances
    float *centers = new float[t];
    float *dists = new float[t];

    // initialize centers and distances
    for (size_t i = 1; i < t + 1; i++)
    {
        centers[i - 1] = (2 * i - 1) * (n / (2 * t));
        dists[i - 1] = 0.0;
    }

    // sort array
    sort(array, array + n);

    double T = 0.0;

    // calling cluster function 10 times
    for (int i = 0; i < 10; i++)
    {
        omp_set_num_threads(t); // set number of threads
        auto start = high_resolution_clock::now(); // start timer
        cluster(n, t, array, centers, dists); // call cluster function
        auto end = high_resolution_clock::now(); // end timer
        T += duration_cast<duration<double, std::milli>>(end - start).count(); // add time to T
    }

    float max_dist = 0.0; // initialize max distance
    size_t t_id = 0; // initialize thread id

    // update max distance and thread id
    for (size_t i = 0; i < t; i++)
    {
        if (max_dist < dists[i])
        {
            max_dist = dists[i];
            t_id = i;
        }
    }

    cout <<max_dist<<endl; // print max distance
    cout <<t_id<<endl; // print thread id
    cout <<T/10<<endl; // print average time

    return 0;
}
