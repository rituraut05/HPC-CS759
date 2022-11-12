#include <random>
#include <iostream>
#include <time.h>
#include <chrono>
#include <ratio>
#include <cmath>
#include "montecarlo.h"
#include <algorithm>
#include <cassert>
using namespace std;
using namespace chrono;
/*
 Q2. In this task,  you will implement an estimation ofπusing the Monte Carlo Method1.The idea is to generatenrandom
 floats in the range [−r, r], whereris the radius of a circle, andcount the number of floats that reside in the
 circle (call itincircle), then use4 * incircle/nas the estimation ofπ.  Whennbecomes very large, this estimation
 could be fairly accurate.Upon implementing this method withomp for, you will also compare the performance
 differencewhen thesimddirective is added.

Q2.b)
Your programtask2.cppshould accomplish the following:
•Create and fill withfloat-type random numbers arrayxof lengthn, wherenis thefirst command line argument,
see below.xshould be drawn from the range [-r,r],whereris the circle radius and it can be set to 1.0.
•Create and fill withfloat-type random numbers arrayyof lengthn, wherenis thefirst command line argument,
see below.yshould be drawn from the range [-r,r],whereris the circle radius and it can be set to 1.0.
•Call  themontecarlofunction  that  returns  the  number  of  points  that  reside  in  thecircle.
•Print the estimatedπ.•Print the time taken to run themontecarlofunction inmilliseconds.
•Compile2:g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp-fno-tree-vectorize -march=native -fopt-info-vec
•Run (wherenis a positive integer,tis an integer in the range [1, 10]):./task2 n t
•Example expected output:
  3.1416
  0.352
*/

int main(int argc, char *argv[])
{
  size_t n = atoi(argv[1]);
  size_t t = atoi(argv[2]);

  int r = 1;
  srand(time(NULL));

  // random number generator
  random_device entropy_source;
  mt19937_64 generator(entropy_source());

  // range for random number
  const int min = -r, max = r;

  // random number distribution
  std::uniform_real_distribution<float> dist(min, max);

  // create array of length n
  float *a = new float[n];
  float *b = new float[n];


  // fill array with random numbers
  for (size_t i = 0; i < n; i++)
  {
    a[i] = dist(generator);
    b[i] = dist(generator);
  }

  double time = 0.0;
  double pi;
  omp_set_num_threads(t);

  for (int i = 0; i < 10; i++)
  {
    omp_set_num_threads(t); // set number of threads
    double start = omp_get_wtime();  // start timer
    pi = 4.0 * float(montecarlo(n, a, b, r)) / n; // call montecarlo function
    double end = omp_get_wtime(); // end timer
    time += (end - start) * 1000; // add time to total time
  }

  printf("%f\n", pi); // print pi
  printf("%f\n", time / 10); // print average time
  return 0;
}
