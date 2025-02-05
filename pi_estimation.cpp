//compile options when rocrand is included: hipcc pi_estimation.cpp -o pi_estimate -std=c++17 -lrocrand                                          



#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>


double estimatePi(int numSamples) {
    int insideCircle = 0;

    for (int i = 0; i < numSamples; ++i) {
        double x = static_cast<double>(rand()) / RAND_MAX;
        double y = static_cast<double>(rand()) / RAND_MAX;

        if (x * x + y * y <= 1.0) {
            ++insideCircle;
        }
    }

    return 4.0 * insideCircle / numSamples;
}

__global__ void estimatePiGPU(double *insideCircle, int numSamples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    rocrand_state_xorwow state;
	rocrand_init(123, tid, 0, &state);
    const auto value1 = rocrand(&state);
    const auto value2 = rocrand(&state);

    if (tid < numSamples) {
        double x = static_cast<double>(value1) / UINT32_MAX;
        double y = static_cast<double>(value2) / UINT32_MAX;

        if (x * x + y * y <= 1.0) {
            atomicAdd(insideCircle, 1);
        }
    }
}


int main() {
    srand(static_cast<unsigned int>(time(0)));

    int numSamples = 10000000000; // Number of random samples

    // Estimate Pi using GPU
    double *insideCircle;
    double *h_result;

    h_result = (double *)malloc(numSamples*sizeof(double));

    hipMalloc(&insideCircle, numSamples*sizeof(double));
    hipMemcpy(insideCircle, h_result, numSamples*sizeof(double), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;

    estimatePiGPU<<<gridSize, blockSize>>>(insideCircle, numSamples);
    
    // Wait for the GPU to finish
    hipDeviceSynchronize();

    hipMemcpy(h_result, insideCircle, sizeof(double), hipMemcpyDeviceToHost);

    double piGPU = 4.0 * *h_result / numSamples;
    std::cout << "Estimated value of Pi using gpu: " << piGPU << std::endl;

    // Estimate Pi using CPU
    double pi = estimatePi(numSamples);
    std::cout << "Estimated value of Pi using cpu: " << pi << std::endl;

    hipFree(insideCircle);
    free(h_result);


    return 0;
}
