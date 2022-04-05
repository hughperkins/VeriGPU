#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))

// #include <stdlib.h>
#include <cstdlib>
#include <ostream>

class MemoryInfo
{
public:
    MemoryInfo(size_t pos, size_t size) : pos(pos), size(size) {}
    ~MemoryInfo() {}
    friend std::ostream &operator<<(std::ostream &os, const MemoryInfo &dt);

    // protected:
    size_t pos;
    size_t size;
};

// class GPURuntime() {
// public:
//     GPURuntime();
//     ~GPURuntime();
// };

// class Controller;
void *gpuMalloc(uint32_t requestedBytes);
void gpuCopyToDevice(void *gpuMemPtr, void *srcData, size_t numBytes);
void gpuCopyFromDevice(void *destData, void *gpuMemPtr, size_t numBytes);
void tick();
void gpuCreateContext();
void gpuDestroyContext();
