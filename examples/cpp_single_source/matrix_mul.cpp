#include "gpu_runtime.h"
#include <iostream>
#include <cassert>


__global__ void matrix_mul(
        uint32_t I, uint32_t K, uint32_t J,
        float *m1, float *m2, float *res) {
    for(uint32_t i = 0; i < I; i++) {
        for (uint32_t j = 0; j < J; j++) {
            float v = 0.0;
            for(uint32_t k = 0; k < K; k++) {
                v += m1[i * K + k] * m2[k * J + j];
            }
            res[i * J + j] = v;
        }
    }
}

int main(int argc, char **argv, char **env)
{
    std::cout << "begin" << std::endl;

    gpuCreateContext();

    uint32_t I = 2;
    uint32_t K = 3;
    uint32_t J = 4;

    // m1 is 2 x 3
    float m1[] = {
        4.2, 3.5, -8.1,
        10.0, 0.3, -0.1
    };
    // m2 is 3 x 4, so 12 values
    float m2[] = {
        3.33, 1.23, 5.1, -0.6,
        +0.1, 0.2, 0.3, 0.4,
        -1.0, -15.0, -0.8, 12.0
    };
    // result will be 2 x 4, i.e. length 8
    float res[8];

    void *ptrGpuM1 = gpuMalloc(I * K * sizeof(float));
    void *ptrGpuM2 = gpuMalloc(K * J * sizeof(float));
    void *ptrGpuRes = gpuMalloc(I * J * sizeof(float));

    gpuCopyToDevice(ptrGpuM1, m1, I * K * sizeof(float));
    gpuCopyToDevice(ptrGpuM2, m2, K * J * sizeof(float));

    // launch the kernel :)
    // remember: single source :) Hopefully we can handle this :)
    matrix_mul<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(
        I, K, J,
        (float *)ptrGpuM1, (float *)ptrGpuM2, (float *)ptrGpuRes);

    gpuCopyFromDevice((void *)&res, ptrGpuRes, I * J * sizeof(float));
    for(int i = 0 ; i < I; i++) {
        for(int j = 0; j < J; j++) {
            std::cout << res[i * J + j] << " ";
        }
        std::cout << std::endl;
    }

    // check what result should be, using pytorch:
    // import torch

    // m1 = torch.tensor([
    //     [4.2, 3.5, -8.1],
    //     [10.0, 0.3, -0.1]
    // ])

    // m2 = torch.tensor([
    //     [3.33, 1.23, 5.1, -0.6],
    //     [0.1, 0.2, 0.3, 0.4],
    //     [-1.0, -15.0, -0.8, 12.0]
    // ])
    // print(m1 @ m2)

    // result should be:
    // tensor([ [ 22.4360, 127.3660, 28.9500, -98.3200 ],
    //          [ 33.4300, 13.8600, 51.1700, -7.0800 ] ])

    float checkValues[] = {22.4360, 127.3660, 28.9500, -98.3200, 33.4300, 13.8600, 51.1700, -7.0800};
    for(int i = 0; i < I * J; i++) {
        assert(abs(checkValues[i] - res[i]) < 0.0002);
    }

    gpuDestroyContext();

    exit(EXIT_SUCCESS);
}
