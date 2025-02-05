//program to do a simple vector addition using HIP

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

using namespace std;
#define N 1000000000

__global__ void vecAdd(float* C, float* A, float* B)
{
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}


int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    
    //allocate memory on the host
    A=(float*)malloc(N*sizeof(float));
    B=(float*)malloc(N*sizeof(float));
    C=(float*)malloc(N*sizeof(float));

    //allocate memory on the device
    hipMalloc(&d_A, N*sizeof(float));
    hipMalloc(&d_B, N*sizeof(float));
    hipMalloc(&d_C, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Copy the input vectors A and B in host memory to the GPU memory
    hipMemcpy(d_A, A, N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, N*sizeof(float), hipMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize=256;
    gridSize=(N+blockSize-1)/blockSize;

    vecAdd<<<gridSize, blockSize>>>(d_C, d_A, d_B);
    // Wait for the GPU to finish
    hipDeviceSynchronize();

    
    // Copy the results in GPU memory back to the CPU
    hipMemcpy(C, d_C, N*sizeof(float), hipMemcpyDeviceToHost);

    //output of the first 10 elements of the array

    for (int i = 0; i < 10; i++) {
        cout << "A[" << i << "] = " << A[i] << '\n';
        cout << "B[" << i << "] = " << B[i] << '\n';
        cout << "C[" << i << "] = " << C[i] << '\n';
    }

    hipFree(A);
    hipFree(B);
    hipFree(C);

    free(A);
    free(B);
    free(C);

    return 0;

}
