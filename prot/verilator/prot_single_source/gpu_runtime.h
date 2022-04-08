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

struct dim3
{
    dim3(unsigned int x, unsigned y, unsigned int z) : x(x), y(y), z(z) {}
    dim3(unsigned int x, unsigned y) : x(x), y(y), z(1) {}
    dim3(unsigned int x) : x(x), y(1), z(1) {}
    dim3() : x(1), y(1), z(1) {}
    // unsigned int pad;
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

std::ostream &operator<<(std::ostream &os, const dim3 &value);
std::ostream &operator<<(std::ostream &os, const size_t value[3]);

// extern "C"
// {
//     int cudaConfigureCall(const dim3 grid, const dim3 block, long long shared = 0, char *stream = 0);
// }
