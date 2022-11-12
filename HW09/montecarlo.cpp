#include <omp.h>
using namespace std;

#include <cstddef>
// this function returns the number of points that lay inside
// a circle using OpenMP parallel for.
// You also need to use the simd directive.

// x - an array of random floats in the range [-radius, radius] with length n.
// y - another array of random floats in the range [-radius, radius] with length n.

int montecarlo(const size_t n, const float *x, const float *y, const float radius)
{
    int sum = 0;

#pragma omp parallel for simd reduction(+ \
                                        : sum)
    for (size_t i = 0; i < n; i++)
    {
        if (x[i] * x[i] + y[i] * y[i] < radius * radius) // check if point is inside circle
        {
            sum++;
        }
        {
            sum += 1;
        }
    }

    return sum;
}