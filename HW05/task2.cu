// #include <cstdio>
// #include <iostream>
// #include <fstream>
// #include <sstream>
// #include <math.h>
// #include <random>

// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #include "matmul.cuh"

// int main(int argc, char* argv[])
// {
//     unsigned int n;
//     unsigned int block_dim;
//     std::istringstream input_1(argv[1]);
//     std::istringstream input_2(argv[2]);
//     if (input_1 >> n && input_2>>block_dim && input_2.eof()) {
//         //create arrays on managed mem
//         int* A, * B, * C;
//         cudaMallocManaged(&A, sizeof(int) * n * n);
//         cudaMallocManaged(&B, sizeof(int) * n * n);
//         cudaMallocManaged(&C, sizeof(int) * n * n);

//         std::random_device entropy_source;
//         std::mt19937 generator(entropy_source());
//         std::uniform_int_distribution<int> dist1(-10,10);
//         for (unsigned int i = 0; i < n*n; i++) {
//             // A[i] = dist1(generator);
//             // B[i] = dist1(generator);
//             A[i] = i;
//             B[i] = 1;
//             C[i] = 0;

//         }

//         // kernel call
//         cudaEvent_t startEvent, stopEvent;
//         cudaEventCreate(&startEvent);
//         cudaEventCreate(&stopEvent);
//         cudaEventRecord(startEvent, 0);

//         matmul_1(A, B, C, n, block_dim);

//         cudaEventRecord(stopEvent, 0);
//         cudaEventSynchronize(stopEvent);
//         float elapsedTime;
//         cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
//         cudaEventDestroy(startEvent);
//         cudaEventDestroy(stopEvent);

//         std::cout << C[0] << "\n";
//         std::cout << C[n * n - 1] << "\n";
//         std::cout << elapsedTime << "\n";

//         cudaFree(A);
//         cudaFree(B);
//         cudaFree(C);
        
//         //Float
//         float *Af, *Bf, *Cf;
//         cudaMallocManaged(&Af, sizeof(float) * n * n);
//         cudaMallocManaged(&Bf, sizeof(float) * n * n);
//         cudaMallocManaged(&Cf, sizeof(float) * n * n);

//         std::uniform_real_distribution<float> dist2(-1.0, 1.0);
//         for (unsigned int i = 0; i < n * n; i++) {
//             // Af[i] = dist2(generator);
//             // Bf[i] = dist2(generator);
//             Af[i] = i;
//             Bf[i] = 1;
//             Cf[i] = 0;
//         }

//         cudaEvent_t startEvent2, stopEvent2;
//         cudaEventCreate(&startEvent2);
//         cudaEventCreate(&stopEvent2);
//         cudaEventRecord(startEvent2, 0);

//         matmul_2(Af, Bf, Cf, n, block_dim);

//         cudaEventRecord(stopEvent2, 0);
//         cudaEventSynchronize(stopEvent2);
//         float elapsedTime2;
//         cudaEventElapsedTime(&elapsedTime2, startEvent2, stopEvent2);
//         cudaEventDestroy(startEvent2);
//         cudaEventDestroy(stopEvent2);

//         std::cout << Cf[0] << "\n";
//         std::cout << Cf[n * n - 1] << "\n";
//         std::cout << elapsedTime2 << "\n";

//         cudaFree(Af);
//         cudaFree(Bf);
//         cudaFree(Cf);

//         //Double
//         double *Ad, *Bd, *Cd;
//         cudaMallocManaged(&Ad, sizeof(double) * n * n);
//         cudaMallocManaged(&Bd, sizeof(double) * n * n);
//         cudaMallocManaged(&Cd, sizeof(double) * n * n);

//         std::uniform_real_distribution<double> dist3(-1.0, 1.0);
//         for (unsigned int i = 0; i < n * n; i++) {
//             // Ad[i] = dist3(generator);
//             // Bd[i] = dist3(generator);
//             Ad[i] = i;
//             Bd[i] = 1;
//             Cd[i] = 0;

//         }

//         // kernel call
//         cudaEvent_t startEvent3, stopEvent3;
//         cudaEventCreate(&startEvent3);
//         cudaEventCreate(&stopEvent3);
//         cudaEventRecord(startEvent3, 0);

//         matmul_3(Ad, Bd, Cd, n, block_dim);

//         cudaEventRecord(stopEvent3, 0);
//         cudaEventSynchronize(stopEvent3);
//         float elapsedTime3;
//         cudaEventElapsedTime(&elapsedTime3, startEvent3, stopEvent3);
//         cudaEventDestroy(startEvent3);
//         cudaEventDestroy(stopEvent3);

//         std::cout << Cd[0] << "\n";
//         std::cout << Cd[n * n - 1] << "\n";
//         std::cout << elapsedTime3 << "\n";

//         cudaFree(Ad);
//         cudaFree(Bd);
//         cudaFree(Cd);
        
//     }

//     return 0;
// }


#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matmul.cuh"

using namespace std;
int main(int argc, char* argv[])
{
    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);
        int* hostA = new int[n*n]; // The A matrix
        int* hostB = new int[n*n];// The B matrix
        int* hostC = new int[n*n];// The output C matrix
        int* deviceA;
        int* deviceB;
        int* deviceC;
        cudaMalloc(&deviceA, sizeof(int) * n * n);
        cudaMalloc(&deviceB, sizeof(int) * n * n);
        cudaMalloc(&deviceC, sizeof(int) * n * n);

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> disInt(-10, 10);

        for (unsigned int i = 0; i < n * n; i++) {
            hostA[i] = disInt(gen);
            hostB[i] = disInt(gen);
            // hostA[i] = 1;
            // hostB[i] = 1;
            hostC[i] = 0;
        }

        cudaMemcpy(deviceA, hostA, sizeof(int) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB, sizeof(int) * n * n, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        matmul_1(deviceA, deviceB, deviceC, n, block_dim);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(hostC, deviceC, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
        cout << hostC[0] << endl;
        cout << hostC[n * n - 1] << endl;
        cout << elapsedTime << endl;

        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);

        delete[] hostA;
        delete[] hostB;
        delete[] hostC;

        ///////////////////Float///////////////////////

        float* hostAf = new float[n*n]; // The A matrix
        float* hostBf = new float[n*n];// The B matrix
        float* hostCf = new float[n*n];// The output C matrix
        float* deviceAf;
        float* deviceBf;
        float* deviceCf;
        cudaMalloc(&deviceAf, sizeof(float) * n * n);
        cudaMalloc(&deviceBf, sizeof(float) * n * n);
        cudaMalloc(&deviceCf, sizeof(float) * n * n);


        uniform_real_distribution<float> disFloat(-1.0, 1.0);

        for (unsigned int i = 0; i < n * n; i++) {
            hostAf[i] = disFloat(gen);
            hostBf[i] = disFloat(gen);
            // hostAf[i] = 1;
            // hostBf[i] = 1;
            hostCf[i] = 0;
        }

        cudaMemcpy(deviceAf, hostAf, sizeof(float) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceBf, hostBf, sizeof(float) * n * n, cudaMemcpyHostToDevice);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        matmul_2(deviceAf, deviceBf, deviceCf, n, block_dim);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(hostCf, deviceCf, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
        cout << hostCf[0] << endl;
        cout << hostCf[n * n - 1] << endl;
        cout << elapsedTime << endl;

        cudaFree(deviceAf);
        cudaFree(deviceBf);
        cudaFree(deviceCf);

        delete[] hostAf;
        delete[] hostBf;
        delete[] hostCf;


    ///////////////////Double///////////////////////

        double* hostAd = new double[n*n]; // The A matrix
        double* hostBd = new double[n*n];// The B matrix
        double* hostCd = new double[n*n];// The output C matrix
        double* deviceAd;
        double* deviceBd;
        double* deviceCd;
        cudaMalloc(&deviceAd, sizeof(double) * n * n);
        cudaMalloc(&deviceBd, sizeof(double) * n * n);
        cudaMalloc(&deviceCd, sizeof(double) * n * n);


        uniform_real_distribution<double> disDouble(-1.0, 1.0);

        for (unsigned int i = 0; i < n * n; i++) {
            hostAd[i] = disDouble(gen);
            hostBd[i] = disDouble(gen);
            // hostAd[i] = 1;
            // hostBd[i] = 1;
            hostCd[i] = 0;
        }

        cudaMemcpy(deviceAd, hostAd, sizeof(double) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceBd, hostBd, sizeof(double) * n * n, cudaMemcpyHostToDevice);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        matmul_3(deviceAd, deviceBd, deviceCd, n, block_dim);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(hostCd, deviceCd, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
        cout << hostCd[0] << endl;
        cout << hostCd[n * n - 1] << endl;
        cout << elapsedTime << endl;

        cudaFree(deviceAd);
        cudaFree(deviceBd);
        cudaFree(deviceCd);

        delete[] hostAd;
        delete[] hostBd;
        delete[] hostCd;
    return 0;
}