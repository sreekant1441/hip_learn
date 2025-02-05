//program to call a simple kernel using hip

#include <iostream>
#include <hip/hip_runtime.h>

using namespace std;

__global__ void hello_gpu() {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    printf("Hello from Navi32!\n");
    printf("my block id is %d\n", hipBlockIdx_x);
    printf("my block dimension is %d\n", hipBlockDim_x);
    printf("My thread id is %d\n", i);
}


int main() {
    hello_gpu<<<1, 1>>>(); // 1 block, 1 threads
    hipDeviceSynchronize();

    return 0;
}
