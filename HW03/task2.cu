/* Q2.  Write a C++ program using CUDA in a file called task2.cuwhich does the following:
•From the host, allocates an array of 16ints on the device calleddA.
•Launches a kernel with 2 blocks, each block having 8 threads.
•Each thread computesax+yand writes the result in one distinct entry of thedAarray.  
Here,–xis the thread’sthreadIdx;–yis the thread’sblockIdx;–ais an integer argument that the kernel takes (so all threads use the samea).  
You needto generatearandomly and then call the kernel with it.
It is up to you how you generatethis random number, one possible approach is described here BestPractice.
•Copies back the data stored in the device arraydAinto a host array calledhA.
•Prints (from the host) the 16 values stored in the host array separated by a single space each.
How to go about it, and what the expected output looks like:
•Compile:nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -otask2
•Run (on Euler, use Slurm sbatch!):./task2
•Expected  output  (followed  by  newline;  yours  could  be  different  depending  on  the  randomnumber generation):
0 10 20 30 40 50 60 70 1 11 21 31 41 51 61 71
*/
#include <iostream>
#include <random>
using namespace std;

// Kernel function to save the integer in the right position of the array
__global__ void kernel(int *dA, int a) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[x + y * 8] = a * x + y;
}

int main() {
    int *dA; // Pointer to the device array
    int hA[16]; // Array on the host

    // Random integer generator
	random_device entropy_source;
	mt19937_64 generator(entropy_source()); 

    // Range for the random number generator = [0,10]
	const int min = 0, max = 10; 
    
    // Uniform distribution to pick the random integer
	uniform_int_distribution<> dist(min, max);
	
	// random integer generator sets the value of a
	int a = dist(generator);

    // Allocate memory for 16 ints on the device
    cudaMalloc(&dA, 16 * sizeof(int));

    // Launch the kernel
    kernel<<<2, 8>>>(dA, a);

    // Copy the data back to the host
    cudaMemcpy(hA, dA, 16 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the array from the host
    for (int i = 0; i < 16; i++) {
        cout << hA[i] << " ";
    }
    cout <<endl;

    // Free the memory on device
    cudaFree(dA);
    return 0;
}