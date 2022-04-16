// things necessary to compile single-source code, at least as far as IR

#include <ostream>

#pragma once

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

extern "C"
{
    int cudaConfigureCall(const dim3 grid, const dim3 block, long long shared = 0, char *stream = 0);
}
