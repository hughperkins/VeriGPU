#pragma once

// #include "cocl/cocl_attributes.h"

#include <cstdint>
#include <cstddef>
#include <iostream>

// #define type1(NAME, TYPE)                        \
//     struct NAME##1                               \
//     {                                            \
//     public:                                      \
//         __devicehost__ NAME##1() {}              \
//         __devicehost__ NAME##1(TYPE x) : x(x) {} \
//         TYPE x;                                  \
//     };

// #define type2(NAME, TYPE)                                      \
//     struct NAME##2                                             \
//     {                                                          \
//     public:                                                    \
//         __devicehost__ NAME##2() {}                            \
//         __devicehost__ NAME##2(TYPE x, TYPE y) : x(x), y(y) {} \
//         TYPE x;                                                \
//         TYPE y;                                                \
//     };

// #define type3(NAME, TYPE)                                                            \
//     struct NAME##3                                                                   \
//     {                                                                                \
//     public:                                                                          \
//         __devicehost__ NAME##3() {}                                                  \
//         __devicehost__ NAME##3(TYPE x, TYPE y, TYPE z, TYPE w) : x(x), y(y), z(z) {} \
//         TYPE x;                                                                      \
//         TYPE y;                                                                      \
//         TYPE z;                                                                      \
//     };

// #define type4(NAME, TYPE)                                                                  \
//     struct NAME##4                                                                         \
//     {                                                                                      \
//     public:                                                                                \
//         __devicehost__ NAME##4() {}                                                        \
//         __devicehost__ NAME##4(TYPE x, TYPE y, TYPE z, TYPE w) : x(x), y(y), z(z), w(w) {} \
//         TYPE x;                                                                            \
//         TYPE y;                                                                            \
//         TYPE z;                                                                            \
//         TYPE w;                                                                            \
//     };

// #define type_all_lengths(NAME, TYPE) \
//     type1(NAME, TYPE)                \
//         type2(NAME, TYPE)            \
//             type3(NAME, TYPE)        \
//                 type4(NAME, TYPE)

// type_all_lengths(uchar, unsigned char)
// type_all_lengths(char, char)
// type_all_lengths(ushort, unsigned short)
// type_all_lengths(short, short)
// type_all_lengths(uint, unsigned int)
// type_all_lengths(int, int)
// type_all_lengths(ulong, uint32_t)
// type_all_lengths(long, int32_t)
// type_all_lengths(ulonglong, uint64_t)
// type_all_lengths(longlong, int64_t)
// type_all_lengths(float, float)
// type_all_lengths(double, double)

// type4(ushort, unsigned short)
// type4(int, int)
// type4(float, float)

// __attribute__((device)) __attribute__((host)) float4 make_float4(float x, float y, float z, float w);
// __attribute__((device)) __attribute__((host)) float2 make_float2(float x, float y);
// __attribute__((device)) __attribute__((host)) double2 make_double2(double x, double y);

#ifdef __CUDACC__
                                                __device__ float4 make_float4(float x, float y, float z, float w);
__device__ float2 make_float2(float x, float y);
__device__ double2 make_double2(double x, double y);
#endif

// float4 make_float4(float x, float y, float z, float w);
// float2 make_float2(float x, float y);
// double2 make_double2(double x, double y);

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
