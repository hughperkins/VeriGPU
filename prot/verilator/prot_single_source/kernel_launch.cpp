// copied from https://github.com/hughperkins/coriander, 8 april 2022

#include "kernel_launch_ext.h"
#include "kernel_launch.h"
// #include "cocl/VERIGPU_memory.h"
// #include "cocl/VERIGPU_clsources.h"
// #include "cocl/VERIGPU_streams.h"
// #include "cocl/VERIGPU_funcs.h"

#include <sys/stat.h>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <stdexcept>
#include <set>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <fstream>
#include <cassert>
#include <experimental/filesystem>

// #include "EasyCL/EasyCL.h"
#include "stringhelper.h"
#include "gpu_runtime.h"

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

#define VERIGPU_SPAM_KERNELLAUNCH

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

static std::string makePreferred(std::string path) {
    return std::experimental::filesystem::path(path).make_preferred();
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
    size_t assemble(uint32_t offset, std::string assembly, vector<unsigned int> &codeWords) {
        // now we have to assemble it...
        // perhaps we can just use clang to assemble it???
        // anyway for now, first try using assembler.py
        // first we should write the assembler somewhere
        char *pString = std::getenv("VERIGPUDIR");
        if (pString == 0)
        {
            std::cout << "ERROR: You need to define VERIGPUDIR" << std::endl;
            throw std::runtime_error("ERROR: You need to define VERIGPUDIR");
        }
        std::string verigpuDir = pString;
        std::cout << "VERIGPUDIR=" << verigpuDir << std::endl;

        std::ofstream of;
        std::string asmPath = makePreferred(verigpuDir + "/build/prog.asm");
        std::cout << "asmPath " << asmPath << std::endl;
        of.open(asmPath);
        of << assembly << std::endl;
        of.close();
        int ret = system(("python " + makePreferred(verigpuDir + "/verigpu/assembler.py") +
                          " --in-asm " + makePreferred(verigpuDir + "/build/prog.asm") +
                          " --out-hex " + makePreferred(verigpuDir + "/build/prog.hex") +
                          " --offset " + std::to_string(offset)
        ).c_str());
        assert(ret == 0);

        // now get size of file
        std::ifstream fin;
        fin.open(makePreferred(verigpuDir + "/build/prog.hex"));
        int numLines = 0;
        // std::string _;
        unsigned int codeWord;
        codeWords.clear();
        // std::vector<unsigned int> codeWords;
        // while(std::getline(fin, _)) {
        while (fin >> hex >> codeWord)
        {
            numLines++;
            codeWords.push_back(codeWord);
        }
        fin.close();
        std::cout << numLines << " lines in hex file" << std::endl;
        size_t kernelCodeSpaceNeeded = numLines << 2;
        std::cout << "kernelCodeSpaceNeeded=" << kernelCodeSpaceNeeded << " bytes" << std::endl;
        std::cout << "len(codeWords)" << codeWords.size() << std::endl;
        return kernelCodeSpaceNeeded;
    }

    std::string Int32Arg::str()
    {
        ostringstream oss;
        oss << "Int32Arg v=" << v;
        return oss.str();
    }

    // std::string Int64Arg::str()
    // {
    //     ostringstream oss;
    //     oss << "Int64Ar v=" << v;
    //     return oss.str();
    // }

    // int32_t getNumCachedKernels()
    // {
    //     return getThreadVars()->getContext()->kernelCache.size();
    // }

    // int32_t getNumKernelCalls()
    // {
    //     return getThreadVars()->getContext()->numKernelCalls;
    // }
} // namespace VeriGPU

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

void setKernelArgPointerVoid(void *ptrVoid)
{
    // This adds a gpu buffer to the kernel args, adding it to the list of unique clmems,
    // if not already present, and adding the offset, as a kernel parameter
    //
    // The size of the buffer is not needed (though the virtual memory system knows it :-) )
    // The elementSize used to be used, but is no longer used/needed. Should probably be
    // removed from the method parameters at some point.

    std::cout << "setKernelArgPointerVoid ptrVoid " << (size_t)ptrVoid << std::endl;
    launchConfiguration.args.push_back(std::unique_ptr<Arg>(new PointerVoidArg(ptrVoid)));
    // pthread_mutex_unlock(&launchMutex);
}

// void setKernelArgInt64(int64_t value)
// {
//     // std::lock_guard<std::recursive_mutex> guard(launchMutex);
//     // pthread_mutex_lock(&launchMutex);
//     launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int64Arg(value)));
//     VERIGPU_PRINT("setKernelArgInt64 " << value);
//     // pthread_mutex_unlock(&launchMutex);
// }

void setKernelArgInt32(int value)
{
    // std::lock_guard<std::recursive_mutex> guard(launchMutex);
    // pthread_mutex_lock(&launchMutex);
    launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int32Arg(value)));
    VERIGPU_PRINT("setKernelArgInt32 " << value);
    // pthread_mutex_unlock(&launchMutex);
}

// void setKernelArgInt8(char value)
// {
//     // std::lock_guard<std::recursive_mutex> guard(launchMutex);
//     // pthread_mutex_lock(&launchMutex);
//     launchConfiguration.args.push_back(std::unique_ptr<Arg>(new Int8Arg(value)));
//     VERIGPU_PRINT("setKernelArgInt8 " << value);
//     // pthread_mutex_unlock(&launchMutex);
// }

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
        VERIGPU_PRINT("kernelGo() kernel: " << launchConfiguration.kernelName);
        std::cout << "kernel source code " << launchConfiguration.deviceriscvsourcecode << std::endl;

        // things we have to do:
        // - allocate memory for stack (note: we should probably just keep this memory around between calls :) )
        // - add header to assmebly which:
        //    - populates stack pointer
        //    - populates a0, a1 etc with function parameters
        //    - calls the function
        //    - halts
        // - note: in reality, we should not be assembling for every kernel launch
        //   TODO: fix this :)
        // - compile assembly, to get size (might want to pass in offset > 2048, to ensure large enough,
        //   since LI becomes one instruction for < 2048, and two instructions otherwise)
        // - allocate gpu buffer for our assembly
        // - recompile assembly with correct offset :P (ideally we could combine the 1st adn third steps somehow)
        // - copy assembly to the gpu
        // - modify comp.sv / proc.sv to have a way of controlling it
        //    - a way to change PC
        //    - a way to enable/disable
        //    - a way to reset (but with a new pc)
        // - modify memory so that both proc and controller can read/write to it
        //   - but for now, we could just have the controller disable proc whilst copying data into memory :)
        //   - still needs a way for them both to interface with memory controller though...
        //   - maybe need to change protocol to talk with memory controller?
        //        - send req=1
        //        - mem controller responds busy=1 or ack=1
        //        - if ack, can send data, once per clock cycle, until put req back to 0
        //        - for receiving,
        //        - hmmm, let's use AXI for that, eg https://developer.arm.com/documentation/102202/0200/Channel-transfers-and-transactions
        //        - but lets use something simpler for now, to get single source compilation and launch working
        //        - so for now we'll just have two write ports on memory, one for controller, and one for proc...
        //          (fix this later)
        // - modify controller.sv to integrate with comp.sv
        // - trigger kernel launch
        // - (for now) wait for kernel to finish (later on, kernel launch will be asynchronous)

        // for comp.sv/proc.sv, so we can have:
        // - enable/disable
        // - add synchronous clear (and, do we even need async reset for the cores?
        //   whats the worst they can do? write to global memory?
        //   probalby need async reset on the controller though I guess?)
        // - whilst disabled, can have something like req_set_pc, and pc_value

        void *stack = gpuMalloc(stackSize);
        std::cout << "allocated stack pos=" << (size_t)(stack) << std::endl;
/*
Example of header for assembly:
li sp, 1000
addi sp, sp, -64
li a0, 5
sw a0, 60(sp)
li a0, 12
sw a0, 56(sp)
li a0, 9
sw a0, 52(sp)
addi a0, sp, 52
li a1, 3
addi a2, sp, 48

jal x1, _Z8sum_intsPjjS_

halt
*/

        std::ostringstream asmHeader;
        asmHeader << "li sp, " << ((size_t)(stack) + stackSize) << std::endl;
        for (int i = 0; i < launchConfiguration.args.size(); i++)
        {
            VERIGPU_PRINT("arg i=" << i << " " << launchConfiguration.args[i]->str());
            //     launchConfiguration.args[i]->inject(kernel);
            launchConfiguration.args[i]->appendToAsmSStream(asmHeader, i);
        }
        asmHeader << "jal x1, " << launchConfiguration.kernelName << std::endl;
        asmHeader << "halt" << std::endl;
        std::cout << "asmHeader:" << std::endl;
        std::cout << asmHeader.str() << std::endl;

        std::string fullAssembly = asmHeader.str() + launchConfiguration.deviceriscvsourcecode;
        std::cout << std::endl;
        std::cout << "fullAssembly" << std::endl;
        std::cout << fullAssembly << std::endl;

        vector<unsigned int> codeWords;
            size_t kernelCodeSpaceNeeded = assemble(0, fullAssembly, codeWords);

        void *gpuKernelSpace = gpuMalloc(kernelCodeSpaceNeeded);

        // now we need to reassemble, with offset at this new position
        assemble((size_t)gpuKernelSpace, fullAssembly, codeWords);
        std::cout << "reassembler at offset " << (size_t)gpuKernelSpace << std::endl;

        gpuCopyToDevice(gpuKernelSpace, &codeWords[0], codeWords.size() << 2);
        std::cout << "copied kernel to device" << std::endl;

        size_t global[3];
        for (int i = 0; i < 3; i++)
        {
            global[i] = launchConfiguration.grid[i] * launchConfiguration.block[i];
        }
        VERIGPU_PRINT("grid: " << launchConfiguration.grid << " block: " << launchConfiguration.block
                            << " global: " << global);
        int workgroupSize = launchConfiguration.block[0] * launchConfiguration.block[1] * launchConfiguration.block[2];
        VERIGPU_PRINT("workgroupSize=" << workgroupSize);

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
