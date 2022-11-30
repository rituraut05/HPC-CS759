/*
Write a program task1.cpp that will accomplish the following:
Create  and  fill  avec vof  length n with datattype  values  generated  any  way  you like 
(with this freedom, it is your responsibility to prevent data overflow);n is the first command 
line argument of this script.
Do the following for eachoptimizeXfunction:
–Call youroptimizeXfunction to get the result ofOPoperations and save it indest.
–Print the result ofdest.
–Print the time taken to run theoptimizeXfunction inmilliseconds.
Compile1:g++ task1.cpp optimize.cpp -Wall -O3 -std=c++17 -o task1 -fno-tree-vectorize
Run on a Euler compute node with aSlurmscript (wherenis a positive integer):./task1 n
*/

#include "optimize.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <random>


using namespace std;
using namespace chrono;

int main(int argc, char * argv[]){
    int n = atoi(argv[1]);
    vec vector = vec(n);
    vector.data = new data_t[n];
    srand(time(NULL));

    // random number generator
    random_device entropy_source;
    mt19937_64 generator(entropy_source());

    // range for random number
    const int min = 0, max = n;

    uniform_int_distribution<int> dist(min, max);

    for (int i = 0; i < n; i++)
    {
        vector.data[i] = dist(generator);
    }

    data_t dest;

    // optimize1
    auto start = high_resolution_clock::now();
    optimize1(&vector, &dest);
    auto end = high_resolution_clock::now();
    cout << dest << endl;
    cout <<duration_cast<duration<double, milli>>(end - start).count() << endl;


    // optimize2
    start = high_resolution_clock::now();
    optimize2(&vector, &dest);
    end = high_resolution_clock::now();
    cout << dest << endl;
    cout << duration_cast<duration<double, milli>>(end - start).count() << endl;

    // optimize3
    start = high_resolution_clock::now();
    optimize3(&vector, &dest);
    end = high_resolution_clock::now();
    cout << dest << endl;
    cout << duration_cast<duration<double, milli>>(end - start).count() << endl;

    // optimize4
    start = high_resolution_clock::now();
    optimize4(&vector, &dest);
    end = high_resolution_clock::now();
    cout << dest << endl;
    cout << duration_cast<duration<double, milli>>(end - start).count() << endl;

    // optimize5
    start = high_resolution_clock::now();
    optimize5(&vector, &dest);
    end = high_resolution_clock::now();
    cout << dest << endl;
    cout << duration_cast<duration<double, milli>>(end - start).count() << endl;

    // float T = 0.0;
    // for(int i=0; i<10; i++){
    //     auto start = high_resolution_clock::now();
    //     optimize1(&vector, &dest);
    //     auto end = high_resolution_clock::now();
    //     T += duration_cast<duration<double, milli>>(end - start).count();
    // }
    // cout << T/10 << endl;

    // T = 0.0;
    // for(int i=0; i<10; i++){
    //     auto start = high_resolution_clock::now();
    //     optimize2(&vector, &dest);
    //     auto end = high_resolution_clock::now();
    //     T += duration_cast<duration<double, milli>>(end - start).count();
    // }
    // cout << T/10 << endl;

    // T = 0.0;
    // for(int i=0; i<10; i++){
    //     auto start = high_resolution_clock::now();
    //     optimize3(&vector, &dest);
    //     auto end = high_resolution_clock::now();
    //     T += duration_cast<duration<double, milli>>(end - start).count();
    // }
    // cout << T/10 << endl;


    // T = 0.0;
    // for(int i=0; i<10; i++){
    //     auto start = high_resolution_clock::now();
    //     optimize4(&vector, &dest);
    //     auto end = high_resolution_clock::now();
    //     T += duration_cast<duration<double, milli>>(end - start).count();
    // }
    // cout << T/10 << endl;


    // T = 0.0;
    // for(int i=0; i<10; i++){
    //     auto start = high_resolution_clock::now();
    //     optimize5(&vector, &dest);
    //     auto end = high_resolution_clock::now();
    //     T += duration_cast<duration<double, milli>>(end - start).count();
    // }
    // cout << T/10 << endl;

    return 0;
}
