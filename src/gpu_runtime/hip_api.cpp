// initially just create stubs for everything.

// (initially created by verigpu/stub_hip_api.py)

#include <iostream>
#include <stdio.h>
#include <cassert>
#include "hip/hip_runtime_api_shortened.h"

// error_t enum from
// https://github.com/ROCm-Developer-Tools/HIP/blob/042c5ee5e618e6582ff87ccae989e85ed3864f76/include/hip/hip_runtime_api.h#L231-L345
// which was under MIT license
typedef enum __HIP_NODISCARD hipError_t
{
    hipSuccess = 0,           ///< Successful completion.
    hipErrorInvalidValue = 1, ///< One or more of the parameters passed to the API call is NULL
                              ///< or not in an acceptable range.
    hipErrorOutOfMemory = 2,
    // Deprecated
    hipErrorMemoryAllocation = 2, ///< Memory allocation error.
    hipErrorNotInitialized = 3,
    // Deprecated
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,
    hipErrorInvalidPitchValue = 12,
    hipErrorInvalidSymbol = 13,
    hipErrorInvalidDevicePointer = 17,   ///< Invalid Device Pointer
    hipErrorInvalidMemcpyDirection = 21, ///< Invalid memory copy direction
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,
    hipErrorNoDevice = 100,      ///< Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice = 101, ///< DeviceID must be in range 0...#compute-devices.
    hipErrorInvalidImage = 200,
    hipErrorInvalidContext = 201, ///< Produced when input context is invalid.
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    // Deprecated
    hipErrorMapBufferObjectFailed = 205, ///< Produced when the IPC memory attach failed from ROCr.
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,
    hipErrorContextAlreadyInUse = 216,
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218, ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    hipErrorFileNotFound = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    hipErrorOperatingSystem = 304,
    hipErrorInvalidHandle = 400,
    // Deprecated
    hipErrorInvalidResourceHandle = 400, ///< Resource handle (hipEvent_t or hipStream_t) invalid.
    hipErrorIllegalState = 401,          ///< Resource required is not in a valid state to perform operation.
    hipErrorNotFound = 500,
    hipErrorNotReady = 600, ///< Indicates that asynchronous operations enqueued earlier are not
                            ///< ready.  This is not actually an error, but is used to distinguish
                            ///< from hipSuccess (which indicates completion).  APIs that return
                            ///< this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701, ///< Out of resources error.
    hipErrorLaunchTimeOut = 702,
    hipErrorPeerAccessAlreadyEnabled =
        704, ///< Peer access was already enabled from the current device.
    hipErrorPeerAccessNotEnabled =
        705, ///< Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess = 708,
    hipErrorContextIsDestroyed = 709,
    hipErrorAssert = 710, ///< Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered =
        712, ///< Produced when trying to lock a page-locked memory.
    hipErrorHostMemoryNotRegistered =
        713, ///< Produced when trying to unlock a non-page-locked memory.
    hipErrorLaunchFailure =
        719, ///< An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge =
        720,                                ///< This error indicates that the number of blocks launched per grid for a kernel
                                            ///< that was launched via cooperative launch APIs exceeds the maximum number of
                                            ///< allowed blocks for the current device
    hipErrorNotSupported = 801,             ///< Produced when the hip API is not supported/implemented
    hipErrorStreamCaptureUnsupported = 900, ///< The operation is not permitted when the stream
                                            ///< is capturing.
    hipErrorStreamCaptureInvalidated = 901, ///< The current capture sequence on the stream
                                            ///< has been invalidated due to a previous error.
    hipErrorStreamCaptureMerge = 902,       ///< The operation would have resulted in a merge of
                                            ///< two independent capture sequences.
    hipErrorStreamCaptureUnmatched = 903,   ///< The capture was not initiated in this stream.
    hipErrorStreamCaptureUnjoined = 904,    ///< The capture sequence contains a fork that was not
                                            ///< joined to the primary stream.
    hipErrorStreamCaptureIsolation = 905,   ///< A dependency would have been created which crosses
                                            ///< the capture sequence boundary. Only implicit
                                            ///< in-stream ordering dependencies  are allowed
                                            ///< to cross the boundary
    hipErrorStreamCaptureImplicit = 906,    ///< The operation would have resulted in a disallowed
                                            ///< implicit dependency on a current capture sequence
                                            ///< from hipStreamLegacy.
    hipErrorCapturedEvent = 907,            ///< The operation is not permitted on an event which was last
                                            ///< recorded in a capturing stream.
    hipErrorStreamCaptureWrongThread = 908, ///< A stream capture sequence not initiated with
                                            ///< the hipStreamCaptureModeRelaxed argument to
                                            ///< hipStreamBeginCapture was passed to
                                            ///< hipStreamEndCapture in a different thread.
    hipErrorGraphExecUpdateFailure = 910,   ///< This error indicates that the graph update
                                            ///< not performed because it included changes which
                                            ///< violated constraintsspecific to instantiated graph
                                            ///< update.
    hipErrorUnknown = 999,                  //< Unknown error.
    // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory = 1052, ///< HSA runtime memory call returned error.  Typically not seen
                                  ///< in production systems.
    hipErrorRuntimeOther = 1053,  ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
    hipErrorTbd                   ///< Marker that more error codes are needed.
} hipError_t;

// various enums and structures, obtained under MIT license from
// https://github.com/ROCm-Developer-Tools/HIP/blob/042c5ee5e618e6582ff87ccae989e85ed3864f76/include/hip/hip_runtime_api.h#L80-L205
//---
// Common headers for both NVCC and HCC paths:

/**
 * hipDeviceProp
 *
 */
typedef struct hipDeviceProp_t
{
    char name[256];                               ///< Device name.
    size_t totalGlobalMem;                        ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;                     ///< Size of shared memory region (in bytes).
    int regsPerBlock;                             ///< Registers per block.
    int warpSize;                                 ///< Warp size.
    int maxThreadsPerBlock;                       ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];                         ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];                           ///< Max grid dimensions (XYZ).
    int clockRate;                                ///< Max clock frequency of the multiProcessors in khz.
    int memoryClockRate;                          ///< Max global memory clock frequency in khz.
    int memoryBusWidth;                           ///< Global memory bus width in bits.
    size_t totalConstMem;                         ///< Size of shared memory region (in bytes).
    int major;                                    ///< Major compute capability.  On HCC, this is an approximation and features may
                                                  ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                                                  ///< feature caps.
    int minor;                                    ///< Minor compute capability.  On HCC, this is an approximation and features may
                                                  ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                                                  ///< feature caps.
    int multiProcessorCount;                      ///< Number of multi-processors (compute units).
    int l2CacheSize;                              ///< L2 cache size.
    int maxThreadsPerMultiProcessor;              ///< Maximum resident threads per multi-processor.
    int computeMode;                              ///< Compute mode.
    int clockInstructionRate;                     ///< Frequency in khz of the timer used by the device-side "clock*"
                                                  ///< instructions.  New for HIP.
    hipDeviceArch_t arch;                         ///< Architectural feature flags.  New for HIP.
    int concurrentKernels;                        ///< Device can possibly execute multiple kernels concurrently.
    int pciDomainID;                              ///< PCI Domain ID
    int pciBusID;                                 ///< PCI Bus ID.
    int pciDeviceID;                              ///< PCI Device ID.
    size_t maxSharedMemoryPerMultiProcessor;      ///< Maximum Shared Memory Per Multiprocessor.
    int isMultiGpuBoard;                          ///< 1 if device is on a multi-GPU board, 0 if not.
    int canMapHostMemory;                         ///< Check whether HIP can map host memory
    int gcnArch;                                  ///< DEPRECATED: use gcnArchName instead
    char gcnArchName[256];                        ///< AMD GCN Arch Name.
    int integrated;                               ///< APU vs dGPU
    int cooperativeLaunch;                        ///< HIP device supports cooperative launch
    int cooperativeMultiDeviceLaunch;             ///< HIP device supports cooperative launch on multiple devices
    int maxTexture1DLinear;                       ///< Maximum size for 1D textures bound to linear memory
    int maxTexture1D;                             ///< Maximum number of elements in 1D images
    int maxTexture2D[2];                          ///< Maximum dimensions (width, height) of 2D images, in image elements
    int maxTexture3D[3];                          ///< Maximum dimensions (width, height, depth) of 3D images, in image elements
    unsigned int *hdpMemFlushCntl;                ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    unsigned int *hdpRegFlushCntl;                ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    size_t memPitch;                              ///< Maximum pitch in bytes allowed by memory copies
    size_t textureAlignment;                      ///< Alignment requirement for textures
    size_t texturePitchAlignment;                 ///< Pitch alignment requirement for texture references bound to pitched memory
    int kernelExecTimeoutEnabled;                 ///< Run time limit for kernels executed on the device
    int ECCEnabled;                               ///< Device has ECC support enabled
    int tccDriver;                                ///< 1:If device is Tesla device using TCC driver, else 0
    int cooperativeMultiDeviceUnmatchedFunc;      ///< HIP device supports cooperative launch on multiple
                                                  /// devices with unmatched functions
    int cooperativeMultiDeviceUnmatchedGridDim;   ///< HIP device supports cooperative launch on multiple
                                                  /// devices with unmatched grid dimensions
    int cooperativeMultiDeviceUnmatchedBlockDim;  ///< HIP device supports cooperative launch on multiple
                                                  /// devices with unmatched block dimensions
    int cooperativeMultiDeviceUnmatchedSharedMem; ///< HIP device supports cooperative launch on multiple
                                                  /// devices with unmatched shared memories
    int isLargeBar;                               ///< 1: if it is a large PCI bar device, else 0
    int asicRevision;                             ///< Revision of the GPU in this device
    int managedMemory;                            ///< Device supports allocating managed memory on this system
    int directManagedMemAccessFromHost;           ///< Host can directly access managed memory on the device without migration
    int concurrentManagedAccess;                  ///< Device can coherently access managed memory concurrently with the CPU
    int pageableMemoryAccess;                     ///< Device supports coherently accessing pageable memory
                                                  ///< without calling hipHostRegister on it
    int pageableMemoryAccessUsesHostPageTables;   ///< Device accesses pageable memory via the host's page tables
} hipDeviceProp_t;

/**
 * Memory type (for pointer attributes)
 */
typedef enum hipMemoryType
{
    hipMemoryTypeHost,   ///< Memory is physically located on host
    hipMemoryTypeDevice, ///< Memory is physically located on device. (see deviceId for specific
                         ///< device)
    hipMemoryTypeArray,  ///< Array memory, physically located on device. (see deviceId for specific
                         ///< device)
    hipMemoryTypeUnified ///< Not used currently
} hipMemoryType;

/**
 * @brief hipKernelNodeAttrID
 * @enum
 *
 */
typedef enum hipKernelNodeAttrID
{
    hipKernelNodeAttributeAccessPolicyWindow = 1,
    hipKernelNodeAttributeCooperative = 2,
} hipKernelNodeAttrID;
typedef enum hipAccessProperty
{
    hipAccessPropertyNormal = 0,
    hipAccessPropertyStreaming = 1,
    hipAccessPropertyPersisting = 2,
} hipAccessProperty;
typedef struct hipAccessPolicyWindow
{
    void *base_ptr;
    hipAccessProperty hitProp;
    float hitRatio;
    hipAccessProperty missProp;
    size_t num_bytes;
} hipAccessPolicyWindow;
typedef union hipKernelNodeAttrValue
{
    hipAccessPolicyWindow accessPolicyWindow;
    int cooperative;
} hipKernelNodeAttrValue;

/**
 * Pointer attributes
 */
typedef struct hipPointerAttribute_t
{
    enum hipMemoryType memoryType;
    int device;
    void *devicePointer;
    void *hostPointer;
    int isManaged;
    unsigned allocationFlags; /* flags specified when memory was allocated*/
    /* peers? */
} hipPointerAttribute_t;

const int coresPerComputeUnit = 1;
// const char *name = "VeriGPU OpenSource GPU";
// const char *gcnArchName = "VeriGPU architecture";

extern "C"
{
    hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId)
    {
        std::cout << "hipGetDeviceProperties" << std::endl;
        std::assert(deviceId == 0); // placeholder, for now...
        sprintf(prop->name, "VeriGPU opensource GPU");
        // prop->name = name;
        prop->totalGlobalMem = 4096;
        prop->sharedMemPerBlock = 0;
        prop->registersPerBlock = 32 * coresPerComputeUnit;
        prop->warpSize = coresPerComputeUnit;
        prop->maxThreadsPerBlock = coresPerComputeUnit;
        sprintf(prop->gcnArchName, "VeriGPU architecture");
        // prop->gcnArchName = gcnArchName;
        prop->canMapHostMemory = false;
        prop->isMultiGpuBoard = false;
        prop->concurrentKernels = 1;
        prop->arch = 0;
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
    uint32_t hipGetLastError()
    {
        std::cout << "hipGetLastError" << std::endl;
        return 0;
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
    void hipGetDevice()
    {
        std::cout << "hipGetDevice" << std::endl;
    }
    void hipGetErrorName()
    {
        std::cout << "hipGetErrorName" << std::endl;
    }
    void hipGetErrorString()
    {
        std::cout << "hipGetErrorString" << std::endl;
    }
    void hipSetDevice()
    {
        std::cout << "hipSetDevice" << std::endl;
    }
    void hipStreamSynchronize()
    {
        std::cout << "hipStreamSynchronize" << std::endl;
    }
    void hipMalloc()
    {
        std::cout << "hipMalloc" << std::endl;
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
        std::cout << "__hipRegisterVar" << std::endl;
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
    uint32_t hipInit()
    {
        std::cout << "hipInit" << std::endl;
        return 0;
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
