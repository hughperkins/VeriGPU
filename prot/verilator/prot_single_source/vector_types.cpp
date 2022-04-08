#include "vector_types.h"

#include <iostream>

// float4 make_float4(float x, float y, float z, float w)
// {
//     return float4(x, y, z, w);
// }

// float2 make_float2(float x, float y)
// {
//     return float2(x, y);
// }

// double2 make_double2(double x, double y)
// {
//     return double2(x, y);
// }

std::ostream &operator<<(std::ostream &os, const dim3 &value)
{
    os << "dim3{" << value.x << "," << value.y << "," << value.z << "}";
    return os;
}

std::ostream &operator<<(std::ostream &os, const size_t value[3])
{
    os << "dim3{" << value[0] << "," << value[1] << "," << value[2] << "}";
    return os;
}
