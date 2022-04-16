#include "gpu_runtime.h"
#include <iostream>
#include <cassert>

__global__ void sum_ints(unsigned  int *in, unsigned int numInts, unsigned int *p_out) {
    // sum the ints in in, and write the result to *out
    // we assume just a single thread/core for now
    unsigned int out = 0;
    for(unsigned int i = 0; i < numInts; i++) {
        out += in[i];
    }
    *p_out = out;
}

int main(int argc, char **argv, char **env)
{
    std::cout << "begin" << std::endl;

    gpuCreateContext();

    uint32_t values[] = {3,1,7,9,11};
    uint32_t numValues = 5;

    void *ptrGpuIn = gpuMalloc(numValues * sizeof(uint32_t));
    void *ptrGpuOut = gpuMalloc(1 * sizeof(uint32_t));

    gpuCopyToDevice(ptrGpuIn, values, numValues * sizeof(uint32_t));
    uint32_t returnValue;

    for(int i = 0; i < 10; i++) {
        // launch the kernel :)
        sum_ints<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>((unsigned int *)ptrGpuIn, numValues, (unsigned int *)ptrGpuOut);

        gpuCopyFromDevice((void *)&returnValue, ptrGpuOut, 1 * sizeof(uint32_t));
        std::cout << "sum_ints.cpp returned result " << returnValue << std::endl;
        assert(returnValue == 31);
    }

    gpuDestroyContext();

    exit(EXIT_SUCCESS);
}
