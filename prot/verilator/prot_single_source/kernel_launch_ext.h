// Copyright Hugh Perkins 2016

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Things we want client code to be able to include, without having to include
// eg llvm 4.0

#pragma once

// #include "cocl/vector_types.h"

#include <memory>
#include <cstdint>

#include "gpu_runtime.h"

// #include "EasyCL/EasyCL.h"

// #include "cocl/cocl_defs.h"

// #define __launch_bounds__(x) __attribute__((launch_bounds(x)))
// #define __launch_bounds__(x)
// #define __launch_bounds__(x, y)

// found this at https://gist.github.com/eliben/b014ac17cbe5a452803f
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

namespace cocl
{
    int32_t getNumCachedKernels(); // this should be per-context or something, though right now, it is not yet
    int32_t getNumKernelCalls();
}

extern "C"
{
    // int cudaConfigureCall(const dim3 grid, const dim3 block, long long shared = 0, char *stream = 0);

    void configureKernel(
        const char *kernelName, const char *llsourcecode);

    size_t cuInit(unsigned int flags);

    void configureKernel(const char *kernelName, const char *devicellsourcecode);
    // void addClmemArg(cl_mem clmem);
    void setKernelArgHostsideBuffer(char *pCpuStruct, int structAllocateSize);
    void setKernelArgGpuBuffer(char *memory_as_charstar, int32_t elementSize);
    void setKernelArgInt64(int64_t value);
    void setKernelArgInt32(int value);
    void setKernelArgInt8(char value);
    void setKernelArgFloat(float value);
    void kernelGo();
}

class ArgStore_base
{
public:
    ArgStore_base()
    {
    }
    virtual ~ArgStore_base()
    {
    }
};

template <typename T>
class ArgStore : public ArgStore_base
{
public:
    ArgStore(T arg) : arg(arg)
    {
    }
    T arg;
};

extern std::unique_ptr<ArgStore_base> g_arg;

#ifdef __CUDACC__
#pragma clang diagnostic ignored "-Wunused-value" // prevents warnings for the triple chevron launch syntax
template <typename T>
size_t cudaSetupArgument(T arg, int flags)
{
    std::cout << "cudaSetupArgument(T arg, int flags) from hostsideopencl_funcs.h" << std::endl;
    g_arg.reset(new ArgStore<T>(arg));
    return 0;
}
template <typename Function, typename Arg1>
void then_execute(Function fn, Arg1 arg1)
{
    fn<<<(32, 1, 1), (32, 1, 1), 0>>>(arg1);
}
template <typename Function, typename Arg1>
size_t cudaLaunch(Function (*fn)(Arg1 arg1))
{
    std::cout << "cudaLaunch(T fn) from hostsideopencl_funcs.h" << std::endl;
    Arg1 arg1 = dynamic_cast<ArgStore<Arg1> *>(g_arg.get())->arg;
    // then_execute<Function,
    fn<<<(32, 1, 1), (32, 1, 1), 0>>>(arg1);
    return 0;
}
#endif // __CUDACC__
