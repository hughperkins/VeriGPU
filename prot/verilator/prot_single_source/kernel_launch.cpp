// copied from https://github.com/hughperkins/coriander, 8 april 2022

#include "kernel_launch_ext.h"
#include "kernel_launch.h"
// #include "cocl/VERIGPU_memory.h"
// #include "cocl/VERIGPU_clsources.h"
// #include "cocl/VERIGPU_streams.h"
// #include "cocl/VERIGPU_funcs.h"

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <mutex>

// #include "EasyCL/EasyCL.h"
#include "stringhelper.h"

// #include "cocl/cocl.h"
// #include "cocl/VERIGPU_memory.h"

// #include "cocl/ir-to-opencl.h"
// #include "cocl/ir-to-opencl-common.h"

// #include "cocl/DebugDumper.h"

using namespace std;
// using namespace easycl;
using namespace VeriGPU;

#ifdef VERIGPU_PRINT
#undef VERIGPU_PRINT
#endif

#ifdef VERIGPU_SPAM_KERNELLAUNCH
#define VERIGPU_PRINT(x) std::cout << "[LAUNCH] " << x << std::endl;
#define WHEN_SPAMMING(x) x
#else
#define VERIGPU_PRINT(x)
#define WHEN_SPAMMING(x)
#endif

extern "C"
{
    void kernel_launch_assure_initialized(void);
}

void kernel_launch_assure_initialized(void)
{
}

namespace VeriGPU
{
    // this lock works as follows:
    // - configureKernel takes out a lock, and does not release it. This first level of locking will remain locked, until
    //   the kernelGo funcgtion is called, and completed
    // - then, each setArg etc method takes out an additional level of locking, in case they are called in parallel
    //   (this second level might not be stricly necessary, I'm not sure...)
    //   this second level is per-call, released at hte end
    // - finally, kernelGo is called, takes out a second-level lock, then at the end releases both levels
    //
    // I suppose we could do this with two standard locks, rather than one recursive one. Might be easier to udnerstand/read
    // in fact
    // std::recursive_mutex launchMutex;
}

using namespace VeriGPU;

static LaunchConfiguration launchConfiguration;
// static DebugDumper debugDumper(&launchConfiguration);

std::unique_ptr<ArgStore_base> g_arg;

size_t cuInit(unsigned int flags)
{
    return 0;
}

int cudaConfigureCall(
    dim3 grid,
    dim3 block, long long sharedMem, char *queue_as_voidstar)
{
    // pthread_mutex_lock(&launchMutex);
    // std::lock_guard< std::recursive_mutex > guard(launchMutex);
    // launchMutex.lock();
    // CoclStream *coclStream = (CoclStream *)queue_as_voidstar;
    // ThreadVars *v = getThreadVars();
    // if (coclStream == 0)
    // {
    //     coclStream = v->currentContext->default_stream.get();
    // }
    // CLQueue *clqueue = coclStream->clqueue;
    if (sharedMem != 0)
    {
        VERIGPU_PRINT("cudaConfigureCall: Not implemented: non-zero shared memory");
        throw runtime_error("cudaConfigureCall: Not implemented: non-zero shared memory");
    }
    int grid_x = grid.x;
    int grid_y = grid.y;
    int grid_z = grid.z;
    int block_x = block.x;
    int block_y = block.y;
    int block_z = block.z;

    VERIGPU_PRINT("cudaConfigureCall(grid=" << grid << ",block=" << block << ",sharedMem=" << sharedMem << ",queue=" << (uint64_t)queue_as_voidstar << ")")

    // launchConfiguration.queue = clqueue;
    // launchConfiguration.coclStream = coclStream;
    launchConfiguration.grid[0] = grid_x;
    launchConfiguration.grid[1] = grid_y;
    launchConfiguration.grid[2] = grid_z;
    launchConfiguration.block[0] = block_x;
    launchConfiguration.block[1] = block_y;
    launchConfiguration.block[2] = block_z;
    return 0;
}

namespace VeriGPU
{

    std::string Int32Arg::str()
    {
        ostringstream oss;
        oss << "Int32Arg=" << v;
        return oss.str();
    }

    std::string Int64Arg::str()
    {
        ostringstream oss;
        oss << "Int64Arg=" << v;
        return oss.str();
    }

    // int32_t getNumCachedKernels()
    // {
    //     return getThreadVars()->getContext()->kernelCache.size();
    // }

    // int32_t getNumKernelCalls()
    // {
    //     return getThreadVars()->getContext()->numKernelCalls;
    // }

    // CLKernel *compileOpenCLKernel(string originalKernelName, string clSourcecode)
    // {
    //     return compileOpenCLKernel(originalKernelName, originalKernelName, originalKernelName, clSourcecode);
    // }

    // CLKernel *compileOpenCLKernel(string originalKernelName, string uniqueKernelName, string shortKernelName, string clSourcecode)
    // {
    //     // returns already-built kernel if available, based on the name
    //     // otherwise builds passed-in clsourcecode, caches that, and returns resulting kernel
    //     // (opencl generation has already happened prior to this function)

    //     ThreadVars *v = getThreadVars();
    //     EasyCL *cl = v->getContext()->getCl();
    //     ofstream f;
    //     v->getContext()->numKernelCalls++;
    //     if (v->getContext()->kernelCache.find(uniqueKernelName) != v->getContext()->kernelCache.end())
    //     {
    //         return v->getContext()->kernelCache[uniqueKernelName];
    //     }
    //     // compile the kernel.  we are still locking the mutex, but I cnat think of a better
    //     // way right now...

    //     string filename = "/tmp/" + easycl::toString(v->getContext()->kernelCache.size()) + ".cl";
    //     if (getenv("VERIGPU_LOAD_CL") != 0)
    //     {
    //         cout << "loading cl sourcecode from " << filename << endl;
    //         ifstream f;
    //         f.open(filename, ios_base::in);
    //         clSourcecode = "";
    //         string line = "";
    //         while (getline(f, line))
    //         {
    //             clSourcecode += line + "\n";
    //         }
    //         f.close();
    //     }
    //     else if (getenv("VERIGPU_DUMP_CL") != 0)
    //     {
    //         cout << "saving cl sourcecode to " << filename << endl;
    //         ofstream f;
    //         f.open(filename, ios_base::out);
    //         f << clSourcecode << endl;
    //         f.close();
    //     }

    //     CLKernel *kernel = 0;
    //     try
    //     {
    //         kernel = cl->buildKernelFromString(clSourcecode, shortKernelName, "", "__internal__", true);
    //         if (getenv("VERIGPU_DUMP_BUILD_LOGS") != 0)
    //         {
    //             if (kernel->buildLog != "")
    //             {
    //                 std::cout << kernel->buildLog << std::endl;
    //             }
    //         }
    //     }
    //     catch (runtime_error &e)
    //     {
    //         cout << "compileOpenCLKernel failed to compile opencl sourcecode" << endl;
    //         cout << "unique kernel name " << uniqueKernelName << endl;
    //         cout << "short kernel name " << shortKernelName << endl;
    //         cout << "writing ll to /tmp/failed-kernel.ll" << endl;

    //         cout << "writing cl to /tmp/failed-kernel.cl" << endl;
    //         f.open("/tmp/failed-kernel.cl", ios_base::out);
    //         f << clSourcecode << endl;
    //         f.close();

    //         throw e;
    //     }
    //     v->getContext()->kernelCache[uniqueKernelName] = kernel;
    //     cl->storeKernel(uniqueKernelName, kernel, true); // this will cause the kernel to be deleted with cl.  Not clean yet, but a start
    //     return kernel;
    // }

    // GenerateOpenCLResult generateOpenCL(
    //     int uniqueClmemCount, std::vector<int> &clmemIndexByClmemArgIndex, string origKernelName, string devicellsourcecode)
    // {
    //     // generates OpenCL source-code, based on passed-in bytecode
    //     // returns cached source-code if available

    //     ThreadVars *v = getThreadVars();

    //     ofstream f;
    //     launchConfiguration.shortKernelName = origKernelName.substr(0, 20);

    //     std::ostringstream uniqueKernelName_ss;
    //     uniqueKernelName_ss << origKernelName;
    //     for (int i = 0; i < clmemIndexByClmemArgIndex.size(); i++)
    //     {
    //         uniqueKernelName_ss << "_" << clmemIndexByClmemArgIndex[i];
    //     }
    //     launchConfiguration.uniqueKernelName = uniqueKernelName_ss.str();
    //     if (v->getContext()->clSourceCodeCache.find(launchConfiguration.uniqueKernelName) != v->getContext()->clSourceCodeCache.end())
    //     {
    //         std::string clSourcecode = v->getContext()->clSourceCodeCache[launchConfiguration.uniqueKernelName];
    //         return GenerateOpenCLResult{clSourcecode, origKernelName, launchConfiguration.shortKernelName, launchConfiguration.uniqueKernelName};
    //     }

    //     // convert to opencl first... based on the kernel name required
    //     try
    //     {
    //         string filename = "/tmp/" + easycl::toString(v->getContext()->clSourceCodeCache.size()) + "-device.ll";
    //         if (getenv("VERIGPU_DUMP_BYTECODE") != 0)
    //         {
    //             cout << "saving deviceside bytecode to " << filename << endl;
    //             ofstream f;
    //             f.open(filename, ios_base::out);
    //             f << devicellsourcecode << endl;
    //             f.close();
    //         }
    //         ModuleClRes res = convertLlStringToCl(
    //             uniqueClmemCount, clmemIndexByClmemArgIndex, devicellsourcecode, origKernelName, launchConfiguration.shortKernelName, v->offsets_32bit);
    //         std::string clSourcecode = res.clSourcecode;
    //         KernelInfo kernelInfo;
    //         kernelInfo.usesVmem = res.usesVmem;
    //         kernelInfo.usesScratch = res.usesScratch;
    //         clSourcecode = "// origKernelName: " + origKernelName + "\n" +
    //                        "// uniqueKernelName: " + launchConfiguration.uniqueKernelName + "\n" +
    //                        "// shortKernelName: " + launchConfiguration.shortKernelName + "\n" +
    //                        "\n" +
    //                        clSourcecode;
    //         v->getContext()->clSourceCodeCache[launchConfiguration.uniqueKernelName] = clSourcecode;
    //         v->getContext()->kernelInfoByUniqueName[launchConfiguration.uniqueKernelName] = kernelInfo;
    //         return GenerateOpenCLResult{clSourcecode, origKernelName, launchConfiguration.shortKernelName, launchConfiguration.uniqueKernelName};
    //     }
    //     catch (runtime_error &e)
    //     {
    //         cout << "generateOpenCL failed to generate opencl sourcecode" << endl;
    //         cout << "kernel name orig=" << origKernelName << endl;
    //         cout << "kernel name short=" << launchConfiguration.shortKernelName << endl;
    //         cout << "kernel name unique=" << launchConfiguration.uniqueKernelName << endl;
    //         cout << "writing ll to /tmp/failed-kernel.ll" << endl;
    //         f.open("/tmp/failed-kernel.ll", ios_base::out);
    //         f << devicellsourcecode << endl;
    //         f.close();
    //         throw e;
    //     }
    // }

} // namespace cocl

void configureKernel(const char *kernelName, const char *deviceriscvsourcecode)
{
    // pthread_mutex_lock(&launchMutex);
    // launchMutex.lock();
    // std::lock_guard<std::recursive_mutex> guard(launchMutex);
    VERIGPU_PRINT("=========================================");
    launchConfiguration.kernelName = kernelName;
    launchConfiguration.deviceriscvsourcecode = deviceriscvsourcecode;

    std::cout << "configureKernel kernelName=" << kernelName << std::endl;

    // in order to handle by-value structs containing pointers to gpu structs, we're first going
    // to add the first Memory object to the clmems, so it is available to the kernel, for
    // dereferencing vmemlocs
    // we are going to assume the first memory is at vmemloc=128 :-). very hacky :-DDD
    // Memory *firstMem = findMemory((const char *)128);

    // we're simply going to assume there is a single memory allocated and take that
    // we'll verify this assumption before launhc, if we are in fact using vmem
    // ThreadVars *v = getThreadVars();
    // Memory *firstMem = *v->getContext()->memories.begin();
    // std::cout << "setKernelArgHostsideBuffer firstMem=" << firstMem << std::endl;
    // if its not zero, then pass it into kernel
    // if (firstMem != 0)
    // {
        // launchConfiguration.clmems.push_back(firstMem->clmem);
        // addClmemArg(firstMem->clmem);
    // }

    // pthread_mutex_unlock(&launchMutex);
}

// void addClmemArg(cl_mem clmem)
// {
//     int clmemIndex = 0;
//     if (launchConfiguration.clmemIndexByClmem.find(clmem) == launchConfiguration.clmemIndexByClmem.end())
//     {
//         clmemIndex = launchConfiguration.clmems.size();
//         launchConfiguration.clmems.push_back(clmem);
//         launchConfiguration.clmemIndexByClmem[clmem] = clmemIndex;
//     }
//     else
//     {
//         clmemIndex = launchConfiguration.clmemIndexByClmem.find(clmem)->second;
//     }
//     launchConfiguration.clmemIndexByClmemArgIndex.push_back(clmemIndex);
// }

void setKernelArgHostsideBuffer(char *pCpuStruct, int structAllocateSize)
{
    // this receives a hostside struct. it will
    // - allocate a gpu buffer, to hold the struct
    // - queue an OpenCL command, to copy the hostside buffer to the gpu buffer
    // - adds the gpu buffer, and its offset, to the kernel parameters:
    //   - add the gpu buffer to list of unique clmems (if not already there)
    //   - records the unique clmem index, for use in generation
    //   - adds an integer arg, with value 0, as the offset arg
    //
    // Things this doesnt do:
    // - parse/walk the struct (thats handled during opencl generation, later on, not here)
    // - (well, and also in patch_hostside, which sends the other pointers through, separately)
    //
    // Things this does definitely need:
    // - struct allocate size, so we know how big to make the gpu buffer, and how much
    //   data to copy across, from the hostside struct pointer location
    // - we wont add the clmem to the virtualmem table, so we wont delegate
    //   anything to the setKernelArgGpuBuffer method (which expects an incoming
    //   pointer to be a virtual pointer, not a cl_mem)

    // std::lock_guard<std::recursive_mutex> guard(launchMutex);
    // pthread_mutex_lock(&launchMutex);
    // ThreadVars *v = getThreadVars();
    // EasyCL *cl = v->getContext()->getCl();
    // cl_context *ctx = cl->context;
    // we're going to:
    // allocate a cl_mem for the struct
    // copy the cpu struct to the cl_mem
    // pass the cl_mem into the kernel

    // we should also:
    // deallocate the cl_mem after calling the kernel
    // (we assume hte struct is passed by-value, so we dont have to actually copy it back afterwards)
    // VERIGPU_PRINT("setKernelArgHostsideBuffer size=" << structAllocateSize);
    // if (structAllocateSize < 4)
    // {
    //     structAllocateSize = 4;
    // }
    // cl_int err;
    // cl_mem gpu_struct = clCreateBuffer(*ctx, CL_MEM_READ_WRITE, structAllocateSize,
    //                                    NULL, &err);
    // EasyCL::checkError(err);
    // err = clEnqueueWriteBuffer(launchConfiguration.queue->queue, gpu_struct, CL_TRUE, 0,
    //                            structAllocateSize, pCpuStruct, 0, NULL, NULL);
    // EasyCL::checkError(err);
    // launchConfiguration.kernelArgsToBeReleased.push_back(gpu_struct);

    // addClmemArg(gpu_struct);

    // int offsetElements = 0;
    // if (v->offsets_32bit)
    // {
    //     launchConfiguration.args.push_back(std::unique_ptr<Arg>(new UInt32Arg((uint32_t)offsetElements)));
    // }
    // else
    // {
    //     launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int64Arg((int64_t)offsetElements)));
    // }

    // pthread_mutex_unlock(&launchMutex);
}

void setKernelArgGpuBuffer(char *memory_as_charstar, int32_t elementSize)
{
    // This adds a gpu buffer to the kernel args, adding it to the list of unique clmems,
    // if not already present, and adding the offset, as a kernel parameter
    //
    // The size of the buffer is not needed (though the virtual memory system knows it :-) )
    // The elementSize used to be used, but is no longer used/needed. Should probably be
    // removed from the method parameters at some point.

    // pthread_mutex_lock(&launchMutex);
    // std::lock_guard<std::recursive_mutex> guard(launchMutex);
    // ThreadVars *v = getThreadVars();

    // Memory *memory = findMemory(memory_as_charstar);
    // if (memory == 0)
    // {
    //     VERIGPU_PRINT("setKernelArgGpuBuffer nullptr");
    //     addClmemArg(0);
    //     if (v->offsets_32bit)
    //     {
    //         launchConfiguration.args.push_back(std::unique_ptr<Arg>(new UInt32Arg(0)));
    //     }
    //     else
    //     {
    //         launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int64Arg(0)));
    //     }
    // }
    // else
    // {
    //     size_t offset = memory->getOffset(memory_as_charstar);
    //     cl_mem clmem = memory->clmem;
    //     // std::cout << " clmem=" << clmem << std::endl;

    //     size_t offsetElements = offset;

    //     VERIGPU_PRINT("setKernelArgGpuBuffer offset=" << offset);

    //     addClmemArg(clmem);

    //     if (v->offsets_32bit)
    //     {
    //         launchConfiguration.args.push_back(std::unique_ptr<Arg>(new UInt32Arg((uint32_t)offsetElements)));
    //     }
    //     else
    //     {
    //         launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int64Arg((int64_t)offsetElements)));
    //     }
    // }
    // pthread_mutex_unlock(&launchMutex);
}

void setKernelArgInt64(int64_t value)
{
    // std::lock_guard<std::recursive_mutex> guard(launchMutex);
    // pthread_mutex_lock(&launchMutex);
    launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int64Arg(value)));
    VERIGPU_PRINT("setKernelArgInt64 " << value);
    // pthread_mutex_unlock(&launchMutex);
}

void setKernelArgInt32(int value)
{
    // std::lock_guard<std::recursive_mutex> guard(launchMutex);
    // pthread_mutex_lock(&launchMutex);
    launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int32Arg(value)));
    VERIGPU_PRINT("setKernelArgInt32 " << value);
    // pthread_mutex_unlock(&launchMutex);
}

void setKernelArgInt8(char value)
{
    // std::lock_guard<std::recursive_mutex> guard(launchMutex);
    // pthread_mutex_lock(&launchMutex);
    launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int8Arg(value)));
    VERIGPU_PRINT("setKernelArgInt8 " << value);
    // pthread_mutex_unlock(&launchMutex);
}

void setKernelArgFloat(float value)
{
    // std::lock_guard<std::recursive_mutex> guard(launchMutex);
    // pthread_mutex_lock(&launchMutex);
    launchConfiguration.args.push_back(std::unique_ptr<Arg>(new FloatArg(value)));
    VERIGPU_PRINT("setKernelArgFloat " << value);
    // pthread_mutex_unlock(&launchMutex);
}

void kernelGo()
{
    try
    {
        // launchMutex.lock();
        // pthread_mutex_lock(&launchMutex);
        // VERIGPU_PRINT("kernelGo queue=" << (void *)launchConfiguration.queue);

        // ThreadVars *v = getThreadVars();

        // GenerateOpenCLResult res = generateOpenCL(
        //     launchConfiguration.clmems.size(), launchConfiguration.clmemIndexByClmemArgIndex, launchConfiguration.kernelName, launchConfiguration.devicellsourcecode);
        VERIGPU_PRINT("kernelGo() kernel: " << launchConfiguration.kernelName);
        std::cout << "kernel source code " << launchConfiguration.deviceriscvsourcecode << std::endl;
        // CLKernel *kernel = compileOpenCLKernel(launchConfiguration.kernelName, res.uniqueKernelName, res.shortKernelName, res.clSourcecode);
        // VERIGPU_PRINT("kernelGo() uniqueKernelName: " << launchConfiguration.uniqueKernelName);

        // KernelInfo kernelInfo = v->getContext()->kernelInfoByUniqueName[launchConfiguration.uniqueKernelName];
        // VERIGPU_PRINT("kernel uses vmem?: " << kernelInfo.usesVmem);
        // VERIGPU_PRINT("kernel uses scratch?: " << kernelInfo.usesScratch);
        // if (kernelInfo.usesVmem)
        // {
        //     if (v->getContext()->memories.size() > 1)
        //     {
        //         std::cout << std::endl;
        //         std::cout << "Error: you are trying to use a kernel that uses double-indirected pointers ('float **' et al)" << std::endl;
        //         std::cout << "whilst you have allocated multiple gpu buffers" << std::endl;
        //         std::cout << std::endl;
        //         std::cout << "This is currently not supported by Coriander" << std::endl;
        //         std::cout << std::endl;
        //         std::cout << "Your options are:" << std::endl;
        //         std::cout << "- update your GPU kernel, to not use double-indirected pointers" << std::endl;
        //         std::cout << "- allocate one single huge GPU memory buffer, instead of many smaller ones" << std::endl;
        //         std::cout << std::endl;
        //         throw std::runtime_error("Error: using vmem with multiple allocations");
        //     }
        //     else
        //     {
        //         WHEN_SPAMMING(Memory *memory = *v->getContext()->memories.begin());
        //         VERIGPU_PRINT("Memory allocation ok: one single allocation at vmem=" << memory->fakePos << " sizeByes=" << memory->bytes);
        //     }
        // }

        // ThreadVars *v = getThreadVars();
        // for (int i = 0; i < launchConfiguration.clmems.size(); i++)
        // {
        //     VERIGPU_PRINT("clmem" << i);
        //     kernel->inout(&launchConfiguration.clmems[i]);
        //     // we also need to write out the offset of this clmem, in our virtual memory system
        //     cl_mem clmem = launchConfiguration.clmems[i];
        //     Memory *memory = findMemoryByClmem(clmem);
        //     uint64_t vmemloc = 0;
        //     if (memory != 0)
        //     { // hostsidegpu buffers will be 0
        //         vmemloc = memory->fakePos;
        //     }
        //     if (v->offsets_32bit)
        //     {
        //         kernel->in((uint32_t)vmemloc);
        //     }
        //     else
        //     {
        //         kernel->in((int64_t)vmemloc);
        //     }
        // }
        // for (int i = 0; i < launchConfiguration.args.size(); i++)
        // {
        //     VERIGPU_PRINT("i=" << i << " " << launchConfiguration.args[i]->str());
        //     launchConfiguration.args[i]->inject(kernel);
        // }

        size_t global[3];
        for (int i = 0; i < 3; i++)
        {
            global[i] = launchConfiguration.grid[i] * launchConfiguration.block[i];
        }
        VERIGPU_PRINT("grid: " << launchConfiguration.grid << " block: " << launchConfiguration.block
                            << " global: " << global);
        int workgroupSize = launchConfiguration.block[0] * launchConfiguration.block[1] * launchConfiguration.block[2];
        VERIGPU_PRINT("workgroupSize=" << workgroupSize);
        // kernel->localInts(max(4, workgroupSize));

        // try
        // {
        //     kernel->run(launchConfiguration.queue, 3, global, launchConfiguration.block);
        // }
        // catch (runtime_error &e)
        // {
        //     if (kernel->buildLog != "")
        //     {
        //         std::cout << kernel->buildLog << std::endl;
        //     }
        //     cout << "kernel failed to run" << endl;
        //     cout << "kernel name: [" << launchConfiguration.kernelName << "]" << endl;
        //     launchMutex.unlock();
        //     launchMutex.unlock();
        //     // pthread_mutex_unlock(&launchMutex);
        //     // pthread_mutex_unlock(&launchMutex);
        //     throw e;
        // }
        // VERIGPU_PRINT(".. kernel queued");
        // cl_int err;
        // err = clFinish(launchConfiguration.queue->queue);
        // EasyCL::checkError(err);
        // debugDumper.maybeDump();

        // for (auto it = launchConfiguration.kernelArgsToBeReleased.begin(); it != launchConfiguration.kernelArgsToBeReleased.end(); it++)
        // {
        //     cl_mem memObject = *it;
        //     err = clReleaseMemObject(memObject);
        //     EasyCL::checkError(err);
        // }
        // launchConfiguration.kernelArgsToBeReleased.clear();
        // launchConfiguration.args.clear();

        // launchConfiguration.clmemIndexByClmem.clear();
        // launchConfiguration.clmems.clear();
        // launchConfiguration.clmemIndexByClmemArgIndex.clear();

        // err = clFinish(launchConfiguration.queue->queue);
        // EasyCL::checkError(err);

        // launchMutex.unlock();
        // launchMutex.unlock();
        // pthread_mutex_unlock(&launchMutex);
        // pthread_mutex_unlock(&launchMutex);
    }
    catch (runtime_error &e)
    {
        std::cout << "caught runtime error " << e.what() << std::endl;
        throw e;
    }
}

// MyClass hostsidefuncs(__FILE__);
