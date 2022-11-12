#include "cluster.h"
#include <cmath>
#include <iostream>

/** 1.  (33pts) In this task,  you are given a function that displays a false sharing issue.
Specifically,in  your  code,  each  thread  will  calculate  the  sum  of  the  distances
between  a  “center  point” associated with this thread and a chunk of entries of a large reference
array arr of size n.  The function declaration is incluster.h; its definition is incluster.cpp.
You will need to use thetechniques discussed in class (Lecture 24) to fix the false sharing issues
and assess any impactof performance (false sharing vs.  no false sharing).  To that end:


1. a)  Modify  the  currentcluster.cppfile  to  solve  the  false  sharing  issue.
In  this  particularproblem, you are allowed to modify the filecluster.cpp, even though it is a provided file.
**/

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists)
{
#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
    dists[tid] = 0.0;
#pragma omp for reduction(+ \
                          : dists[tid])
    for (size_t i = 0; i < n; i++)
    {
      dists[tid] += std::fabs(arr[i] - centers[tid]);
    }
  }
}