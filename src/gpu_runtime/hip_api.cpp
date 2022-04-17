// initially just create stubs for everything.

// (initially created by verigpu/stub_hip_api.py)

#include <iostream>
#include <stdio.h>
#include <cassert>
#include "hip/hip_runtime_api_shortened.h"
#include "gpu_runtime.h"


const int coresPerComputeUnit = 1;

extern "C"
{
    uint32_t hipInit()
    {
        std::cout << "hipInit" << std::endl;
        gpuCreateContext();
        return 0;
    }
    hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId)
    {
        std::cout << "hipGetDeviceProperties" << std::endl;
        assert(deviceId == 0); // placeholder, for now...
        sprintf(prop->name, "VeriGPU opensource GPU");
        // prop->name = name;
        prop->totalGlobalMem = 4096;
        prop->totalConstMem = 0;
        prop->sharedMemPerBlock = 0;
        prop->maxSharedMemoryPerMultiProcessor = 0;
        prop->regsPerBlock = 32 * coresPerComputeUnit;
        prop->warpSize = coresPerComputeUnit;
        prop->maxThreadsPerBlock = coresPerComputeUnit;
        sprintf(prop->gcnArchName, "VeriGPU architecture");
        // prop->gcnArchName = gcnArchName;
        prop->canMapHostMemory = false;
        prop->isMultiGpuBoard = false;
        prop->concurrentKernels = 1;
        // prop->arch = 0;
        prop->l2CacheSize = 0;
        prop->major = 2;
        prop->minor = 0;
        return hipSuccess;
    }
    uint32_t hipGetDeviceCount(int *p_cnt)
    {
        std::cout << "hipGetDeviceCount" << std::endl;
        *p_cnt = 1;
        return 0;
    }
    hipError_t hipGetDevice(int *deviceId)
    {
        std::cout << "hipGetDevice" << std::endl;
        *deviceId = 0;
        return hipSuccess;
    }
    hipError_t hipSetDevice(int deviceId)
    {
        std::cout << "hipSetDevice" << std::endl;
        return hipSuccess;
    }
    uint32_t hipGetLastError()
    {
        std::cout << "hipGetLastError" << std::endl;
        return 0;
    }
    hipError_t hipMalloc(void **ptr, size_t size)
    {
        std::cout << "hipMalloc size=" << size << std::endl;
        *ptr = gpuMalloc(size);
        return hipSuccess;
    }

    // auto-generated stubs:
    void hipMemcpy()
    {
        std::cout << "hipMemcpy" << std::endl;
    }
    void __hipPopCallConfiguration()
    {
        std::cout << "__hipPopCallConfiguration" << std::endl;
    }
    void hipLaunchKernel()
    {
        std::cout << "hipLaunchKernel" << std::endl;
    }
    void __hipRegisterFunction()
    {
        // std::cout << "__hipRegisterFunction" << std::endl;
    }
    void __hipRegisterFatBinary()
    {
        // std::cout << "__hipRegisterFatBinary" << std::endl;
    }
    void __hipUnregisterFatBinary()
    {
        // std::cout << "__hipUnregisterFatBinary" << std::endl;
    }
    void __hipPushCallConfiguration()
    {
        std::cout << "__hipPushCallConfiguration" << std::endl;
    }
    void hipMemsetAsync()
    {
        std::cout << "hipMemsetAsync" << std::endl;
    }
    void hipMemset()
    {
        std::cout << "hipMemset" << std::endl;
    }
    void hipMemcpyAsync()
    {
        std::cout << "hipMemcpyAsync" << std::endl;
    }
    void hipGetErrorName()
    {
        std::cout << "hipGetErrorName" << std::endl;
    }
    void hipGetErrorString()
    {
        std::cout << "hipGetErrorString" << std::endl;
    }
    void hipStreamSynchronize()
    {
        std::cout << "hipStreamSynchronize" << std::endl;
    }
    void hipFree()
    {
        std::cout << "hipFree" << std::endl;
    }
    void hipPointerGetAttributes()
    {
        std::cout << "hipPointerGetAttributes" << std::endl;
    }
    void hipStreamQuery()
    {
        std::cout << "hipStreamQuery" << std::endl;
    }
    void hipMemcpy2DAsync()
    {
        std::cout << "hipMemcpy2DAsync" << std::endl;
    }
    void __gnu_h2f_ieee()
    {
        std::cout << "__gnu_h2f_ieee" << std::endl;
    }
    void hipModuleUnload()
    {
        std::cout << "hipModuleUnload" << std::endl;
    }
    void hipModuleLoad()
    {
        std::cout << "hipModuleLoad" << std::endl;
    }
    void hipModuleLoadData()
    {
        std::cout << "hipModuleLoadData" << std::endl;
    }
    void hipModuleGetFunction()
    {
        std::cout << "hipModuleGetFunction" << std::endl;
    }
    void _Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j()
    {
        std::cout << "_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j" << std::endl;
    }
    void hipEventRecord()
    {
        std::cout << "hipEventRecord" << std::endl;
    }
    void hipDeviceSynchronize()
    {
        std::cout << "hipDeviceSynchronize" << std::endl;
    }
    void hipMallocManaged()
    {
        std::cout << "hipMallocManaged" << std::endl;
    }
    void hipPeekAtLastError()
    {
        std::cout << "hipPeekAtLastError" << std::endl;
    }
    void hipMemcpy2D()
    {
        std::cout << "hipMemcpy2D" << std::endl;
    }
    void hipMemset2DAsync()
    {
        std::cout << "hipMemset2DAsync" << std::endl;
    }
    void hipEventCreate()
    {
        std::cout << "hipEventCreate" << std::endl;
    }
    void hipDeviceGetAttribute()
    {
        std::cout << "hipDeviceGetAttribute" << std::endl;
    }
    void hipEventSynchronize()
    {
        std::cout << "hipEventSynchronize" << std::endl;
    }
    void hipEventElapsedTime()
    {
        std::cout << "hipEventElapsedTime" << std::endl;
    }
    void hipEventDestroy()
    {
        std::cout << "hipEventDestroy" << std::endl;
    }
    void hiprtcCreateProgram()
    {
        std::cout << "hiprtcCreateProgram" << std::endl;
    }
    void hiprtcCompileProgram()
    {
        std::cout << "hiprtcCompileProgram" << std::endl;
    }
    void hiprtcGetCodeSize()
    {
        std::cout << "hiprtcGetCodeSize" << std::endl;
    }
    void hiprtcGetCode()
    {
        std::cout << "hiprtcGetCode" << std::endl;
    }
    void hiprtcDestroyProgram()
    {
        std::cout << "hiprtcDestroyProgram" << std::endl;
    }
    void hiprtcGetProgramLogSize()
    {
        std::cout << "hiprtcGetProgramLogSize" << std::endl;
    }
    void hiprtcGetProgramLog()
    {
        std::cout << "hiprtcGetProgramLog" << std::endl;
    }
    void hipModuleLaunchKernel()
    {
        std::cout << "hipModuleLaunchKernel" << std::endl;
    }
    void hipHostMalloc()
    {
        std::cout << "hipHostMalloc" << std::endl;
    }
    void hipHostFree()
    {
        std::cout << "hipHostFree" << std::endl;
    }
    void hipMemGetInfo()
    {
        std::cout << "hipMemGetInfo" << std::endl;
    }
    void hipDeviceCanAccessPeer()
    {
        std::cout << "hipDeviceCanAccessPeer" << std::endl;
    }
    void hipDeviceEnablePeerAccess()
    {
        std::cout << "hipDeviceEnablePeerAccess" << std::endl;
    }
    void hipStreamDestroy()
    {
        std::cout << "hipStreamDestroy" << std::endl;
    }
    void hipDriverGetVersion()
    {
        std::cout << "hipDriverGetVersion" << std::endl;
    }
    void hipRuntimeGetVersion()
    {
        std::cout << "hipRuntimeGetVersion" << std::endl;
    }
    void hipStreamCreate()
    {
        std::cout << "hipStreamCreate" << std::endl;
    }
    void hipEventCreateWithFlags()
    {
        std::cout << "hipEventCreateWithFlags" << std::endl;
    }
    void hipStreamWaitEvent()
    {
        std::cout << "hipStreamWaitEvent" << std::endl;
    }
    void hipMemcpyToSymbol()
    {
        std::cout << "hipMemcpyToSymbol" << std::endl;
    }
    void hipMemcpyFromSymbol()
    {
        std::cout << "hipMemcpyFromSymbol" << std::endl;
    }
    void __hipRegisterVar()
    {
        // std::cout << "__hipRegisterVar" << std::endl;
    }
    void hipDeviceSetCacheConfig()
    {
        std::cout << "hipDeviceSetCacheConfig" << std::endl;
    }
    void hipStreamCreateWithFlags()
    {
        std::cout << "hipStreamCreateWithFlags" << std::endl;
    }
    void hipExtMallocWithFlags()
    {
        std::cout << "hipExtMallocWithFlags" << std::endl;
    }
    void hipDeviceGetByPCIBusId()
    {
        std::cout << "hipDeviceGetByPCIBusId" << std::endl;
    }
    void hipDeviceGetPCIBusId()
    {
        std::cout << "hipDeviceGetPCIBusId" << std::endl;
    }
    void hipIpcCloseMemHandle()
    {
        std::cout << "hipIpcCloseMemHandle" << std::endl;
    }
    void hipIpcGetMemHandle()
    {
        std::cout << "hipIpcGetMemHandle" << std::endl;
    }
    void hipIpcOpenMemHandle()
    {
        std::cout << "hipIpcOpenMemHandle" << std::endl;
    }
    void hsa_amd_pointer_info()
    {
        std::cout << "hsa_amd_pointer_info" << std::endl;
    }
    void hipHostRegister()
    {
        std::cout << "hipHostRegister" << std::endl;
    }
    void hipHostGetDevicePointer()
    {
        std::cout << "hipHostGetDevicePointer" << std::endl;
    }
    void hipHostUnregister()
    {
        std::cout << "hipHostUnregister" << std::endl;
    }
    void hipFuncGetAttributes()
    {
        std::cout << "hipFuncGetAttributes" << std::endl;
    }
    void hipExtLaunchMultiKernelMultiDevice()
    {
        std::cout << "hipExtLaunchMultiKernelMultiDevice" << std::endl;
    }
    void hsa_system_get_info()
    {
        std::cout << "hsa_system_get_info" << std::endl;
    }
    void hsa_executable_iterate_symbols()
    {
        std::cout << "hsa_executable_iterate_symbols" << std::endl;
    }
    void hsa_amd_memory_pool_allocate()
    {
        std::cout << "hsa_amd_memory_pool_allocate" << std::endl;
    }
    void hsa_queue_load_read_index_relaxed()
    {
        std::cout << "hsa_queue_load_read_index_relaxed" << std::endl;
    }
    void hsa_code_object_reader_create_from_file()
    {
        std::cout << "hsa_code_object_reader_create_from_file" << std::endl;
    }
    void hsa_agent_get_info()
    {
        std::cout << "hsa_agent_get_info" << std::endl;
    }
    void hsa_executable_get_symbol()
    {
        std::cout << "hsa_executable_get_symbol" << std::endl;
    }
    void hsa_amd_agents_allow_access()
    {
        std::cout << "hsa_amd_agents_allow_access" << std::endl;
    }
    void hsa_signal_load_relaxed()
    {
        std::cout << "hsa_signal_load_relaxed" << std::endl;
    }
    void hsa_iterate_agents()
    {
        std::cout << "hsa_iterate_agents" << std::endl;
    }
    void hsa_init()
    {
        std::cout << "hsa_init" << std::endl;
    }
    void hsa_executable_symbol_get_info()
    {
        std::cout << "hsa_executable_symbol_get_info" << std::endl;
    }
    void hsa_amd_profiling_set_profiler_enabled()
    {
        std::cout << "hsa_amd_profiling_set_profiler_enabled" << std::endl;
    }
    void hsa_amd_profiling_get_async_copy_time()
    {
        std::cout << "hsa_amd_profiling_get_async_copy_time" << std::endl;
    }
    void hsa_amd_memory_async_copy()
    {
        std::cout << "hsa_amd_memory_async_copy" << std::endl;
    }
    void hsa_queue_store_write_index_relaxed()
    {
        std::cout << "hsa_queue_store_write_index_relaxed" << std::endl;
    }
    void hsa_signal_wait_scacquire()
    {
        std::cout << "hsa_signal_wait_scacquire" << std::endl;
    }
    void hsa_shut_down()
    {
        std::cout << "hsa_shut_down" << std::endl;
    }
    void hsa_signal_create()
    {
        std::cout << "hsa_signal_create" << std::endl;
    }
    void hsa_executable_freeze()
    {
        std::cout << "hsa_executable_freeze" << std::endl;
    }
    void hsa_signal_store_screlease()
    {
        std::cout << "hsa_signal_store_screlease" << std::endl;
    }
    void hsa_queue_destroy()
    {
        std::cout << "hsa_queue_destroy" << std::endl;
    }
    void hsa_signal_destroy()
    {
        std::cout << "hsa_signal_destroy" << std::endl;
    }
    void hsa_amd_memory_async_copy_rect()
    {
        std::cout << "hsa_amd_memory_async_copy_rect" << std::endl;
    }
    void hsa_executable_load_agent_code_object()
    {
        std::cout << "hsa_executable_load_agent_code_object" << std::endl;
    }
    void hsa_queue_load_write_index_relaxed()
    {
        std::cout << "hsa_queue_load_write_index_relaxed" << std::endl;
    }
    void hsa_executable_create_alt()
    {
        std::cout << "hsa_executable_create_alt" << std::endl;
    }
    void hsa_system_get_major_extension_table()
    {
        std::cout << "hsa_system_get_major_extension_table" << std::endl;
    }
    void hsa_signal_store_relaxed()
    {
        std::cout << "hsa_signal_store_relaxed" << std::endl;
    }
    void hsa_queue_create()
    {
        std::cout << "hsa_queue_create" << std::endl;
    }
    void hsa_amd_agent_iterate_memory_pools()
    {
        std::cout << "hsa_amd_agent_iterate_memory_pools" << std::endl;
    }
    void hsa_amd_signal_async_handler()
    {
        std::cout << "hsa_amd_signal_async_handler" << std::endl;
    }
    void hsa_amd_profiling_get_dispatch_time()
    {
        std::cout << "hsa_amd_profiling_get_dispatch_time" << std::endl;
    }
    void hsa_amd_memory_pool_get_info()
    {
        std::cout << "hsa_amd_memory_pool_get_info" << std::endl;
    }
    void hsa_status_string()
    {
        std::cout << "hsa_status_string" << std::endl;
    }
    void hsa_memory_free()
    {
        std::cout << "hsa_memory_free" << std::endl;
    }
    void hsa_amd_profiling_async_copy_enable()
    {
        std::cout << "hsa_amd_profiling_async_copy_enable" << std::endl;
    }
    void hipCtxGetCurrent()
    {
        std::cout << "hipCtxGetCurrent" << std::endl;
    }
    void hipCtxSetCurrent()
    {
        std::cout << "hipCtxSetCurrent" << std::endl;
    }
    void hipDeviceTotalMem()
    {
        std::cout << "hipDeviceTotalMem" << std::endl;
    }
    void _Z24hipHccModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_()
    {
        std::cout << "_Z24hipHccModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_" << std::endl;
    }
    void hipEventQuery()
    {
        std::cout << "hipEventQuery" << std::endl;
    }
    void hipStreamCreateWithPriority()
    {
        std::cout << "hipStreamCreateWithPriority" << std::endl;
    }
    void hipKernelNameRefByPtr()
    {
        std::cout << "hipKernelNameRefByPtr" << std::endl;
    }
    void hipKernelNameRef()
    {
        std::cout << "hipKernelNameRef" << std::endl;
    }
    void roctxMarkA()
    {
        std::cout << "roctxMarkA" << std::endl;
    }
    void hipMemcpyWithStream()
    {
        std::cout << "hipMemcpyWithStream" << std::endl;
    }
    void hipOccupancyMaxActiveBlocksPerMultiprocessor()
    {
        std::cout << "hipOccupancyMaxActiveBlocksPerMultiprocessor" << std::endl;
    }
    void roctxRangePop()
    {
        std::cout << "roctxRangePop" << std::endl;
    }
    void roctxRangePushA()
    {
        std::cout << "roctxRangePushA" << std::endl;
    }
    void hipProfilerStop()
    {
        std::cout << "hipProfilerStop" << std::endl;
    }
    void hipProfilerStart()
    {
        std::cout << "hipProfilerStart" << std::endl;
    }
    void hipStreamGetPriority()
    {
        std::cout << "hipStreamGetPriority" << std::endl;
    }
    void hipDeviceGetStreamPriorityRange()
    {
        std::cout << "hipDeviceGetStreamPriorityRange" << std::endl;
    }
}
