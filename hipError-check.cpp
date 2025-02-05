/*Allocate a device array that requires twice the number of bytes that are in device global memory
and query the hip return status and hip string*/

#include <hip/hip_runtime.h>
#include <iostream>
using namespace std;

#define HIP_CHECK(error) \
  if (error != hipSuccess) { \
    cout << "Error: " << hipGetErrorString(error) << endl; \
    return 1; \
  }


int main() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);

  cout << "Device name: " << prop.name << endl;
  cout << "Total global memory: " << prop.totalGlobalMem << endl;


  size_t size = 2 * prop.totalGlobalMem;
  void *d_data;
  hipError_t error = hipMalloc(&d_data, size);
  HIP_CHECK(error);



  return 0;
}
