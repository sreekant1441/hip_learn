#include <iostream>
#include <hip/hip_runtime.h>

using namespace std;

__global__ void threed_kernel() {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    printf("X block dimension: %d block id: %d  thread id: %d value: %d\n", blockDim.x,blockIdx.x,threadIdx.x, i);
    printf("Y block dimension: %d block id: %d  thread id: %d value: %d\n", blockDim.y,blockIdx.y,threadIdx.y, j);
    printf("Z block dimension: %d block id: %d  thread id: %d value: %d\n", blockDim.z,blockIdx.z,threadIdx.z, k);   
}


int main() {

    /*  BlockDim.n is the number of threads in a block and must be less than 1024

    
        kernel launch format <<<dim3(n_xblocks, n_yblocks, n_zblocks), dim3(n_threads_xdim, n_threads_ydim, n_threads_zdim)>>>

        Grid Dimensions (dim3(1, 1, 2)):

        dim3(1, 1, 2) specifies the grid dimensions.
        1 in the x-dimension means there is 1 block in the x-direction.
        1 in the y-dimension means there is 1 block in the y-direction.
        2 in the z-dimension means there are 2 blocks in the z-direction.
        This results in a total of 1 * 1 * 2 = 2 blocks in the grid.
        Block Dimensions (dim3(1, 1, 1)):

        dim3(1, 1, 1) specifies the block dimensions.
        1 in the x-dimension means there is 1 thread in the x-direction.
        1 in the y-dimension means there is 1 thread in the y-direction.
        1 in the z-dimension means there is 1 thread in the z-direction.
        This results in a total of 1 * 1 * 1 = 1 thread per block.
    */

    threed_kernel<<<dim3(1, 1, 2), dim3(1, 1, 1)>>>();  
    hipDeviceSynchronize();


    return 0;
}
