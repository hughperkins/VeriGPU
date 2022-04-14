/*
Send some integers to the gpu, launch a kernel to calculate their sum,
receive the result back to cpu-side, check correct

This is a work in progress. Stretch goal. Not working yet.

To run it (once it's working), current plan is to run:

examples/cpp_single_source/sum_floats/run.sh
*/
#include "gpu_runtime.h"
#include <iostream>
#include <cassert>

__global__ void sum_floats(float *in, unsigned int numValues, float *p_out) {
    // sum the ints in in, and write the result to *out
    // we assume just a single thread/core for now
    float out = 0.0;
    for (unsigned int i = 0; i < numValues; i++) {
        out += in[i];
    }
    *p_out = out;
}

int main(int argc, char **argv, char **env)
{
    std::cout << "begin" << std::endl;

    gpuCreateContext();

    float values[] = {4.2, 3.5, 8.1, 100.0, 0.0003};
    uint32_t numValues = 5;

    void *ptrGpuIn = gpuMalloc(numValues * sizeof(float));
    void *ptrGpuOut = gpuMalloc(1 * sizeof(float));

    gpuCopyToDevice(ptrGpuIn, values, numValues * sizeof(float));
    float returnValue;

    // launch the kernel :)
    // remember: single source :) Hopefully we can handle this :)
    sum_floats<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>((float *)ptrGpuIn, numValues, (float *)ptrGpuOut);

    gpuCopyFromDevice((void *)&returnValue, ptrGpuOut, 1 * sizeof(float));
    std::cout << "sum_floats.cpp returned result " << returnValue << std::endl;
    assert(abs(returnValue - 115.8003) < 0.0002); // 115.803

    gpuDestroyContext();

    exit(EXIT_SUCCESS);
}
