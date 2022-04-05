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

    // first copy data in
    // we'll just make up our own commands for now
    uint32_t values[] = {3,1,7,9,11};
    uint32_t numValues = 5;
    std::cout << " before cpuMalloc" << std::endl;
    void *ptrMemory = gpuMalloc(numValues * sizeof(uint32_t));
    std::cout << "found memory at " << ptrMemory << std::endl;

    gpuCopyToDevice(ptrMemory, values, numValues * sizeof(uint32_t));

    uint32_t valuesBack[] = {0, 0, 0, 0, 0};

    gpuCopyFromDevice((void *)valuesBack, ptrMemory, numValues * sizeof(uint32_t));
    // copyToGpu(values, );
    for(int i = 0; i < numValues; i++) {
        std::cout << "c++ received data from gpu i=" << i << " val=" << valuesBack[i] << std::endl;
    }
    for(int i = 0; i < numValues; i++) {
        assert(valuesBack[i] == values[i]);
    }

    gpuDestroyContext();

    exit(EXIT_SUCCESS);
}
