/*a)  Implement  in  a  file  called count.cu the  function count as declared and described in count.cuh.
Your count function should be able to take a thrust::device_vector, for instance, named d_in(filled by integers),
and fill the output values array with the unique integers that appear in d_in in ascending order, as well as
the output counts array with the corresponding occurrences of these integers.  A brief example is shown below:
Example input:din= [3, 5, 1, 2, 3, 1]
Expected output:values= [1, 2, 3, 5]
Expected output:counts= [2, 1, 2, 1]
Hints:Since the length of values and counts may not be equal to the length of d_in, you may want to use
thrust::inner_product to find the number of “jumps” (whena[i-1] !=a[i]) as you step through the sorted array
(the input array is not sorted, so you wouldhave  to do  a sort using Thrust built-in function).
You  can refer to  Lecture 18  and19  for thrust::sortexamples. There  are  other  valid  options  as  well,
for instance,thrust::unique.thrust::reducebykey could be helpful.*/
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include "count.cuh"

using namespace std;

void count(const thrust::device_vector<int> &d_in, thrust::device_vector<int> &values, thrust::device_vector<int> &counts)
{

    // device vectors
    thrust::device_vector<int> d_input(int(d_in.size()));
    thrust::device_vector<int> d_val(int(d_in.size()));

    d_input = d_in;

    // fill d_val with 1s
    thrust::fill(d_val.begin(), d_val.end(), 1);

    // sort the input array
    thrust::sort(d_input.begin(), d_input.end());

    // reduce by key
    auto tail = thrust::reduce_by_key(d_input.begin(), d_input.end(), d_val.begin(), values.begin(), counts.begin());

    // resize the vectors to the size of the output
    values.resize(tail.first - values.begin());
    counts.resize(tail.second - counts.begin());
}
