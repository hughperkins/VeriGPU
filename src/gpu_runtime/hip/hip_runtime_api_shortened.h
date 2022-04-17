// shortened version of file of similar name at
// https://github.com/ROCm-Developer-Tools/HIP/blob/042c5ee5e618e6582ff87ccae989e85ed3864f76/include/hip/hip_runtime_api.h
// provided under MIT license

/*
Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @file hip_runtime_api.h
 *
 * @brief Defines the API signatures for HIP runtime.
 * This file can be compiled with a standard compiler.
 */

#pragma once

// #ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_API_H
// #define HIP_INCLUDE_HIP_HIP_RUNTIME_API_H

#include <string.h> // for getDeviceProp
#include <hip/hip_version.h>
#include <hip/hip_common.h>

enum
{
    HIP_SUCCESS = 0,
    HIP_ERROR_INVALID_VALUE,
    HIP_ERROR_NOT_INITIALIZED,
    HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
};

typedef struct
{
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;    ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1; ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics : 1;    ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1; ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd : 1;        ///< 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1; ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics : 1; ///< 64-bit integer atomics for shared memory.

    // Doubles
    unsigned hasDoubles : 1; ///< Double-precision floating point.

    // Warp cross-lane operations
    unsigned hasWarpVote : 1;    ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot : 1;  ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle : 1; ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift : 1; ///< Funnel two words into one with shift&mask caps.

    // Sync
    unsigned hasThreadFenceSystem : 1; ///< __threadfence_system.
    unsigned hasSyncThreadsExt : 1;    ///< __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    unsigned hasSurfaceFuncs : 1;       ///< Surface functions.
    unsigned has3dGrid : 1;             ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism : 1; ///< Dynamic parallelism.
} hipDeviceArch_t;

typedef struct hipUUID_t
{
    char bytes[16];
} hipUUID;

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

// hack to get these to show up in Doxygen:
/**
 *     @defgroup GlobalDefs Global enum and defines
 *     @{
 *
 */

// Ignoring error-code return values from hip APIs is discouraged. On C++17,
// we can make that yield a warning
#if __cplusplus >= 201703L
#define __HIP_NODISCARD [[nodiscard]]
#else
#define __HIP_NODISCARD
#endif

/*
 * @brief hipError_t
 * @enum
 * @ingroup Enumerations
 */
// Developer note - when updating these, update the hipErrorName and hipErrorString functions in
// NVCC and HCC paths Also update the hipCUDAErrorTohipError function in NVCC path.

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

#undef __HIP_NODISCARD

/*
 * @brief hipDeviceAttribute_t
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipDeviceAttribute_t
{
    hipDeviceAttributeCudaCompatibleBegin = 0,

    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
    hipDeviceAttributeAccessPolicyMaxWindowSize,                          ///< Cuda only. The maximum size of the window policy in bytes.
    hipDeviceAttributeAsyncEngineCount,                                   ///< Cuda only. Asynchronous engines number.
    hipDeviceAttributeCanMapHostMemory,                                   ///< Whether host memory can be mapped into device address space
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,                  ///< Cuda only. Device can access host registered memory
                                                                          ///< at the same virtual address as the CPU
    hipDeviceAttributeClockRate,                                          ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeComputeMode,                                        ///< Compute mode that device is currently in.
    hipDeviceAttributeComputePreemptionSupported,                         ///< Cuda only. Device supports Compute Preemption.
    hipDeviceAttributeConcurrentKernels,                                  ///< Device can possibly execute multiple kernels concurrently.
    hipDeviceAttributeConcurrentManagedAccess,                            ///< Device can coherently access managed memory concurrently with the CPU
    hipDeviceAttributeCooperativeLaunch,                                  ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,                       ///< Support cooperative launch on multiple devices
    hipDeviceAttributeDeviceOverlap,                                      ///< Cuda only. Device can concurrently copy memory and execute a kernel.
                                                                          ///< Deprecated. Use instead asyncEngineCount.
    hipDeviceAttributeDirectManagedMemAccessFromHost,                     ///< Host can directly access managed memory on
                                                                          ///< the device without migration
    hipDeviceAttributeGlobalL1CacheSupported,                             ///< Cuda only. Device supports caching globals in L1
    hipDeviceAttributeHostNativeAtomicSupported,                          ///< Cuda only. Link between the device and the host supports native atomic operations
    hipDeviceAttributeIntegrated,                                         ///< Device is integrated GPU
    hipDeviceAttributeIsMultiGpuBoard,                                    ///< Multiple GPU devices.
    hipDeviceAttributeKernelExecTimeout,                                  ///< Run time limit for kernels executed on the device
    hipDeviceAttributeL2CacheSize,                                        ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    hipDeviceAttributeLocalL1CacheSupported,                              ///< caching locals in L1 is supported
    hipDeviceAttributeLuid,                                               ///< Cuda only. 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeLuidDeviceNodeMask,                                 ///< Cuda only. Luid device node mask. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeComputeCapabilityMajor,                             ///< Major compute capability version number.
    hipDeviceAttributeManagedMemory,                                      ///< Device supports allocating managed memory on this system
    hipDeviceAttributeMaxBlocksPerMultiProcessor,                         ///< Cuda only. Max block size per multiprocessor
    hipDeviceAttributeMaxBlockDimX,                                       ///< Max block size in width.
    hipDeviceAttributeMaxBlockDimY,                                       ///< Max block size in height.
    hipDeviceAttributeMaxBlockDimZ,                                       ///< Max block size in depth.
    hipDeviceAttributeMaxGridDimX,                                        ///< Max grid size  in width.
    hipDeviceAttributeMaxGridDimY,                                        ///< Max grid size  in height.
    hipDeviceAttributeMaxGridDimZ,                                        ///< Max grid size  in depth.
    hipDeviceAttributeMaxSurface1D,                                       ///< Maximum size of 1D surface.
    hipDeviceAttributeMaxSurface1DLayered,                                ///< Cuda only. Maximum dimensions of 1D layered surface.
    hipDeviceAttributeMaxSurface2D,                                       ///< Maximum dimension (width, height) of 2D surface.
    hipDeviceAttributeMaxSurface2DLayered,                                ///< Cuda only. Maximum dimensions of 2D layered surface.
    hipDeviceAttributeMaxSurface3D,                                       ///< Maximum dimension (width, height, depth) of 3D surface.
    hipDeviceAttributeMaxSurfaceCubemap,                                  ///< Cuda only. Maximum dimensions of Cubemap surface.
    hipDeviceAttributeMaxSurfaceCubemapLayered,                           ///< Cuda only. Maximum dimension of Cubemap layered surface.
    hipDeviceAttributeMaxTexture1DWidth,                                  ///< Maximum size of 1D texture.
    hipDeviceAttributeMaxTexture1DLayered,                                ///< Cuda only. Maximum dimensions of 1D layered texture.
    hipDeviceAttributeMaxTexture1DLinear,                                 ///< Maximum number of elements allocatable in a 1D linear texture.
                                                                          ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
    hipDeviceAttributeMaxTexture1DMipmap,                                 ///< Cuda only. Maximum size of 1D mipmapped texture.
    hipDeviceAttributeMaxTexture2DWidth,                                  ///< Maximum dimension width of 2D texture.
    hipDeviceAttributeMaxTexture2DHeight,                                 ///< Maximum dimension hight of 2D texture.
    hipDeviceAttributeMaxTexture2DGather,                                 ///< Cuda only. Maximum dimensions of 2D texture if gather operations  performed.
    hipDeviceAttributeMaxTexture2DLayered,                                ///< Cuda only. Maximum dimensions of 2D layered texture.
    hipDeviceAttributeMaxTexture2DLinear,                                 ///< Cuda only. Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
    hipDeviceAttributeMaxTexture2DMipmap,                                 ///< Cuda only. Maximum dimensions of 2D mipmapped texture.
    hipDeviceAttributeMaxTexture3DWidth,                                  ///< Maximum dimension width of 3D texture.
    hipDeviceAttributeMaxTexture3DHeight,                                 ///< Maximum dimension height of 3D texture.
    hipDeviceAttributeMaxTexture3DDepth,                                  ///< Maximum dimension depth of 3D texture.
    hipDeviceAttributeMaxTexture3DAlt,                                    ///< Cuda only. Maximum dimensions of alternate 3D texture.
    hipDeviceAttributeMaxTextureCubemap,                                  ///< Cuda only. Maximum dimensions of Cubemap texture
    hipDeviceAttributeMaxTextureCubemapLayered,                           ///< Cuda only. Maximum dimensions of Cubemap layered texture.
    hipDeviceAttributeMaxThreadsDim,                                      ///< Maximum dimension of a block
    hipDeviceAttributeMaxThreadsPerBlock,                                 ///< Maximum number of threads per block.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,                        ///< Maximum resident threads per multiprocessor.
    hipDeviceAttributeMaxPitch,                                           ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeMemoryBusWidth,                                     ///< Global memory bus width in bits.
    hipDeviceAttributeMemoryClockRate,                                    ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeComputeCapabilityMinor,                             ///< Minor compute capability version number.
    hipDeviceAttributeMultiGpuBoardGroupID,                               ///< Cuda only. Unique ID of device group on the same multi-GPU board
    hipDeviceAttributeMultiprocessorCount,                                ///< Number of multiprocessors on the device.
    hipDeviceAttributeName,                                               ///< Device name.
    hipDeviceAttributePageableMemoryAccess,                               ///< Device supports coherently accessing pageable memory
                                                                          ///< without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables,             ///< Device accesses pageable memory via the host's page tables
    hipDeviceAttributePciBusId,                                           ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,                                        ///< PCI Device ID.
    hipDeviceAttributePciDomainID,                                        ///< PCI Domain ID.
    hipDeviceAttributePersistingL2CacheMaxSize,                           ///< Cuda11 only. Maximum l2 persisting lines capacity in bytes
    hipDeviceAttributeMaxRegistersPerBlock,                               ///< 32-bit registers available to a thread block. This number is shared
                                                                          ///< by all thread blocks simultaneously resident on a multiprocessor.
    hipDeviceAttributeMaxRegistersPerMultiprocessor,                      ///< 32-bit registers available per block.
    hipDeviceAttributeReservedSharedMemPerBlock,                          ///< Cuda11 only. Shared memory reserved by CUDA driver per block.
    hipDeviceAttributeMaxSharedMemoryPerBlock,                            ///< Maximum shared memory available per block in bytes.
    hipDeviceAttributeSharedMemPerBlockOptin,                             ///< Cuda only. Maximum shared memory per block usable by special opt in.
    hipDeviceAttributeSharedMemPerMultiprocessor,                         ///< Cuda only. Shared memory available per multiprocessor.
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio,                   ///< Cuda only. Performance ratio of single precision to double precision.
    hipDeviceAttributeStreamPrioritiesSupported,                          ///< Cuda only. Whether to support stream priorities.
    hipDeviceAttributeSurfaceAlignment,                                   ///< Cuda only. Alignment requirement for surfaces
    hipDeviceAttributeTccDriver,                                          ///< Cuda only. Whether device is a Tesla device using TCC driver
    hipDeviceAttributeTextureAlignment,                                   ///< Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment,                              ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeTotalConstantMemory,                                ///< Constant memory size in bytes.
    hipDeviceAttributeTotalGlobalMem,                                     ///< Global memory available on devicice.
    hipDeviceAttributeUnifiedAddressing,                                  ///< Cuda only. An unified address space shared with the host.
    hipDeviceAttributeUuid,                                               ///< Cuda only. Unique ID in 16 byte.
    hipDeviceAttributeWarpSize,                                           ///< Warp size in threads.
    hipDeviceAttributeMemoryPoolsSupported,                               ///< Device supports HIP Stream Ordered Memory Allocator

    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,

    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin, ///< Frequency in khz of the timer used by the device-side "clock*"
    hipDeviceAttributeArch,                                                      ///< Device architecture
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,                          ///< Maximum Shared Memory PerMultiprocessor.
    hipDeviceAttributeGcnArch,                                                   ///< Device gcn architecture
    hipDeviceAttributeGcnArchName,                                               ///< Device gcnArch name in 256 bytes
    hipDeviceAttributeHdpMemFlushCntl,                                           ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,                                           ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,                       ///< Supports cooperative launch on multiple
                                                                                 ///< devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,                    ///< Supports cooperative launch on multiple
                                                                                 ///< devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,                   ///< Supports cooperative launch on multiple
                                                                                 ///< devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,                  ///< Supports cooperative launch on multiple
                                                                                 ///< devices with unmatched shared memories
    hipDeviceAttributeIsLargeBar,                                                ///< Whether it is LargeBar
    hipDeviceAttributeAsicRevision,                                              ///< Revision of the GPU in this device
    hipDeviceAttributeCanUseStreamWaitValue,                                     ///< '1' if Device supports hipStreamWaitValue32() and
                                                                                 ///< hipStreamWaitValue64(), '0' otherwise.
    hipDeviceAttributeImageSupport,                                              ///< '1' if Device supports image, '0' otherwise.
    hipDeviceAttributePhysicalMultiProcessorCount,                               ///< All available physical compute
                                                                                 ///< units for the device
    hipDeviceAttributeFineGrainSupport,                                          ///< '1' if Device supports fine grain, '0' otherwise

    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000,
    // Extended attributes for vendors
} hipDeviceAttribute_t;

enum hipComputeMode
{
    hipComputeModeDefault = 0,
    hipComputeModeExclusive = 1,
    hipComputeModeProhibited = 2,
    hipComputeModeExclusiveProcess = 3
};

/**
 * @}
 */

// #if (defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)) && !(defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__))

#include <stdint.h>
#include <stddef.h>
#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif
#include <hip/amd_detail/host_defines.h>
#include <hip/driver_types.h>
#include <hip/texture_types.h>
#include <hip/surface_types.h>
#if defined(_MSC_VER)
#define DEPRECATED(msg) __declspec(deprecated(msg))
#else // !defined(_MSC_VER)
#define DEPRECATED(msg) __attribute__((deprecated(msg)))
#endif // !defined(_MSC_VER)
#define DEPRECATED_MSG "This API is marked as deprecated and may not be supported in future releases. For more details please refer https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_deprecated_api_list.md"
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)
#define HIP_LAUNCH_PARAM_END ((void *)0x03)
#ifdef __cplusplus
#define __dparm(x) \
    = x
#else
#define __dparm(x)
#endif
#ifdef __GNUC__
#pragma GCC visibility push(default)
#endif
#ifdef __cplusplus
namespace hip_impl
{
    hipError_t hip_init();
} // namespace hip_impl
#endif
// Structure definitions:
#ifdef __cplusplus
extern "C"
{
#endif
    //---
    // API-visible structures
    typedef struct ihipCtx_t *hipCtx_t;
    // Note many APIs also use integer deviceIds as an alternative to the device pointer:
    typedef int hipDevice_t;
    typedef enum hipDeviceP2PAttr
    {
        hipDevP2PAttrPerformanceRank = 0,
        hipDevP2PAttrAccessSupported,
        hipDevP2PAttrNativeAtomicSupported,
        hipDevP2PAttrHipArrayAccessSupported
    } hipDeviceP2PAttr;
    typedef struct ihipStream_t *hipStream_t;
#define hipIpcMemLazyEnablePeerAccess 0
#define HIP_IPC_HANDLE_SIZE 64
    typedef struct hipIpcMemHandle_st
    {
        char reserved[HIP_IPC_HANDLE_SIZE];
    } hipIpcMemHandle_t;
    typedef struct hipIpcEventHandle_st
    {
        char reserved[HIP_IPC_HANDLE_SIZE];
    } hipIpcEventHandle_t;
    typedef struct ihipModule_t *hipModule_t;
    typedef struct ihipModuleSymbol_t *hipFunction_t;
    /**
     * HIP memory pool
     */
    typedef struct ihipMemPoolHandle_t *hipMemPool_t;

    typedef struct hipFuncAttributes
    {
        int binaryVersion;
        int cacheModeCA;
        size_t constSizeBytes;
        size_t localSizeBytes;
        int maxDynamicSharedSizeBytes;
        int maxThreadsPerBlock;
        int numRegs;
        int preferredShmemCarveout;
        int ptxVersion;
        size_t sharedSizeBytes;
    } hipFuncAttributes;
    typedef struct ihipEvent_t *hipEvent_t;
    enum hipLimit_t
    {
        hipLimitPrintfFifoSize = 0x01,
        hipLimitMallocHeapSize = 0x02,
    };
/**
 * @addtogroup GlobalDefs More
 * @{
 */
// Flags that can be used with hipStreamCreateWithFlags.
/** Default stream creation flags. These are used with hipStreamCreate().*/
#define hipStreamDefault 0x00

/** Stream does not implicitly synchronize with null stream.*/
#define hipStreamNonBlocking 0x01

// Flags that can be used with hipEventCreateWithFlags.
/** Default flags.*/
#define hipEventDefault 0x0

/** Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.*/
#define hipEventBlockingSync 0x1

/** Disable event's capability to record timing information. May improve performance.*/
#define hipEventDisableTiming 0x2

/** Event can support IPC. Warnig: It is not supported in HIP.*/
#define hipEventInterprocess 0x4

/** Use a device-scope release when recording this event. This flag is useful to obtain more
 * precise timings of commands between events.  The flag is a no-op on CUDA platforms.*/
#define hipEventReleaseToDevice 0x40000000

/** Use a system-scope release when recording this event. This flag is useful to make
 * non-coherent host memory visible to the host. The flag is a no-op on CUDA platforms.*/
#define hipEventReleaseToSystem 0x80000000

// Flags that can be used with hipHostMalloc.
/** Default pinned memory allocation on the host.*/
#define hipHostMallocDefault 0x0

/** Memory is considered allocated by all contexts.*/
#define hipHostMallocPortable 0x1

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #hipHostGetDevicePointer.*/
#define hipHostMallocMapped 0x2

/** Allocates the memory as write-combined. On some system configurations, write-combined allocation
 * may be transferred faster across the PCI Express bus, however, could have low read efficiency by
 * most CPUs. It's a good option for data tranfer from host to device via mapped pinned memory.*/
#define hipHostMallocWriteCombined 0x4

/** Host memory allocation will follow numa policy set by user.*/
#define hipHostMallocNumaUser 0x20000000

/** Allocate coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific allocation.*/
#define hipHostMallocCoherent 0x40000000

/** Allocate non-coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific allocation.*/
#define hipHostMallocNonCoherent 0x80000000

/** Memory can be accessed by any stream on any device*/
#define hipMemAttachGlobal 0x01

/** Memory cannot be accessed by any stream on any device.*/
#define hipMemAttachHost 0x02

/** Memory can only be accessed by a single stream on the associated device.*/
#define hipMemAttachSingle 0x04

#define hipDeviceMallocDefault 0x0

/** Memory is allocated in fine grained region of device.*/
#define hipDeviceMallocFinegrained 0x1

/** Memory represents a HSA signal.*/
#define hipMallocSignalMemory 0x2

// Flags that can be used with hipHostRegister.
/** Memory is Mapped and Portable.*/
#define hipHostRegisterDefault 0x0

/** Memory is considered registered by all contexts.*/
#define hipHostRegisterPortable 0x1

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #hipHostGetDevicePointer.*/
#define hipHostRegisterMapped 0x2

/** Not supported.*/
#define hipHostRegisterIoMemory 0x4

/** Coarse Grained host memory lock.*/
#define hipExtHostRegisterCoarseGrained 0x8

/** Automatically select between Spin and Yield.*/
#define hipDeviceScheduleAuto 0x0

/** Dedicate a CPU core to spin-wait. Provides lowest latency, but burns a CPU core and may
 * consume more power.*/
#define hipDeviceScheduleSpin 0x1

/** Yield the CPU to the operating system when waiting. May increase latency, but lowers power
 * and is friendlier to other threads in the system.*/
#define hipDeviceScheduleYield 0x2
#define hipDeviceScheduleBlockingSync 0x4
#define hipDeviceScheduleMask 0x7
#define hipDeviceMapHost 0x8
#define hipDeviceLmemResizeToMax 0x16
/** Default HIP array allocation flag.*/
#define hipArrayDefault 0x00
#define hipArrayLayered 0x01
#define hipArraySurfaceLoadStore 0x02
#define hipArrayCubemap 0x04
#define hipArrayTextureGather 0x08
#define hipOccupancyDefault 0x00
#define hipCooperativeLaunchMultiDeviceNoPreSync 0x01
#define hipCooperativeLaunchMultiDeviceNoPostSync 0x02
#define hipCpuDeviceId ((int)-1)
#define hipInvalidDeviceId ((int)-2)
// Flags that can be used with hipExtLaunch Set of APIs.
/** AnyOrderLaunch of kernels.*/
#define hipExtAnyOrderLaunch 0x01
// Flags to be used with hipStreamWaitValue32 and hipStreamWaitValue64.
#define hipStreamWaitValueGte 0x0
#define hipStreamWaitValueEq 0x1
#define hipStreamWaitValueAnd 0x2
#define hipStreamWaitValueNor 0x3
// Stream per thread
/** Implicit stream per application thread.*/
#define hipStreamPerThread ((hipStream_t)2)
    /*
     * @brief HIP Memory Advise values
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipMemoryAdvise
    {
        hipMemAdviseSetReadMostly = 1,          ///< Data will mostly be read and only occassionally
                                                ///< be written to
        hipMemAdviseUnsetReadMostly = 2,        ///< Undo the effect of hipMemAdviseSetReadMostly
        hipMemAdviseSetPreferredLocation = 3,   ///< Set the preferred location for the data as
                                                ///< the specified device
        hipMemAdviseUnsetPreferredLocation = 4, ///< Clear the preferred location for the data
        hipMemAdviseSetAccessedBy = 5,          ///< Data will be accessed by the specified device,
                                                ///< so prevent page faults as much as possible
        hipMemAdviseUnsetAccessedBy = 6,        ///< Let HIP to decide on the page faulting policy
                                                ///< for the specified device
        hipMemAdviseSetCoarseGrain = 100,       ///< The default memory model is fine-grain. That allows
                                                ///< coherent operations between host and device, while
                                                ///< executing kernels. The coarse-grain can be used
                                                ///< for data that only needs to be coherent at dispatch
                                                ///< boundaries for better performance
        hipMemAdviseUnsetCoarseGrain = 101      ///< Restores cache coherency policy back to fine-grain
    } hipMemoryAdvise;
    /*
     * @brief HIP Coherency Mode
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipMemRangeCoherencyMode
    {
        hipMemRangeCoherencyModeFineGrain = 0,    ///< Updates to memory with this attribute can be
                                                  ///< done coherently from all devices
        hipMemRangeCoherencyModeCoarseGrain = 1,  ///< Writes to memory with this attribute can be
                                                  ///< performed by a single device at a time
        hipMemRangeCoherencyModeIndeterminate = 2 ///< Memory region queried contains subregions with
                                                  ///< both hipMemRangeCoherencyModeFineGrain and
                                                  ///< hipMemRangeCoherencyModeCoarseGrain attributes
    } hipMemRangeCoherencyMode;
    /*
     * @brief HIP range attributes
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipMemRangeAttribute
    {
        hipMemRangeAttributeReadMostly = 1,           ///< Whether the range will mostly be read and
                                                      ///< only occassionally be written to
        hipMemRangeAttributePreferredLocation = 2,    ///< The preferred location of the range
        hipMemRangeAttributeAccessedBy = 3,           ///< Memory range has hipMemAdviseSetAccessedBy
                                                      ///< set for the specified device
        hipMemRangeAttributeLastPrefetchLocation = 4, ///< The last location to where the range was
                                                      ///< prefetched
        hipMemRangeAttributeCoherencyMode = 100,      ///< Returns coherency mode
                                                      ///< @ref hipMemRangeCoherencyMode for the range
    } hipMemRangeAttribute;

    /**
     * @brief HIP memory pool attributes
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipMemPoolAttr
    {
        /**
         * (value type = int)
         * Allow @p hipMemAllocAsync to use memory asynchronously freed
         * in another streams as long as a stream ordering dependency
         * of the allocating stream on the free action exists.
         * hip events and null stream interactions can create the required
         * stream ordered dependencies. (default enabled)
         */
        hipMemPoolReuseFollowEventDependencies = 0x1,
        /**
         * (value type = int)
         * Allow reuse of already completed frees when there is no dependency
         * between the free and allocation. (default enabled)
         */
        hipMemPoolReuseAllowOpportunistic = 0x2,
        /**
         * (value type = int)
         * Allow @p hipMemAllocAsync to insert new stream dependencies
         * in order to establish the stream ordering required to reuse
         * a piece of memory released by cuFreeAsync (default enabled).
         */
        hipMemPoolReuseAllowInternalDependencies = 0x3,
        /**
         * (value type = uint64_t)
         * Amount of reserved memory in bytes to hold onto before trying
         * to release memory back to the OS. When more than the release
         * threshold bytes of memory are held by the memory pool, the
         * allocator will try to release memory back to the OS on the
         * next call to stream, event or context synchronize. (default 0)
         */
        hipMemPoolAttrReleaseThreshold = 0x4,
        /**
         * (value type = uint64_t)
         * Amount of backing memory currently allocated for the mempool.
         */
        hipMemPoolAttrReservedMemCurrent = 0x5,
        /**
         * (value type = uint64_t)
         * High watermark of backing memory allocated for the mempool since the
         * last time it was reset. High watermark can only be reset to zero.
         */
        hipMemPoolAttrReservedMemHigh = 0x6,
        /**
         * (value type = uint64_t)
         * Amount of memory from the pool that is currently in use by the application.
         */
        hipMemPoolAttrUsedMemCurrent = 0x7,
        /**
         * (value type = uint64_t)
         * High watermark of the amount of memory from the pool that was in use by the application since
         * the last time it was reset. High watermark can only be reset to zero.
         */
        hipMemPoolAttrUsedMemHigh = 0x8
    } hipMemPoolAttr;
    /**
     * @brief Specifies the type of location
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipMemLocationType
    {
        hipMemLocationTypeInvalid = 0,
        hipMemLocationTypeDevice = 1 ///< Device location, thus it's HIP device ID
    } hipMemLocationType;
    /**
     * Specifies a memory location.
     *
     * To specify a gpu, set type = @p hipMemLocationTypeDevice and set id = the gpu's device ID
     */
    typedef struct hipMemLocation
    {
        hipMemLocationType type; ///< Specifies the location type, which describes the meaning of id
        int id;                  ///< Identifier for the provided location type @p hipMemLocationType
    } hipMemLocation;
    /**
     * @brief Specifies the memory protection flags for mapping
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipMemAccessFlags
    {
        hipMemAccessFlagsProtNone = 0,     ///< Default, make the address range not accessible
        hipMemAccessFlagsProtRead = 1,     ///< Set the address range read accessible
        hipMemAccessFlagsProtReadWrite = 3 ///< Set the address range read-write accessible
    } hipMemAccessFlags;
    /**
     * Memory access descriptor
     */
    typedef struct hipMemAccessDesc
    {
        hipMemLocation location; ///< Location on which the accessibility has to change
        hipMemAccessFlags flags; ///< Accessibility flags to set
    } hipMemAccessDesc;
    /**
     * @brief Defines the allocation types
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipMemAllocationType
    {
        hipMemAllocationTypeInvalid = 0x0,
        /** This allocation type is 'pinned', i.e. cannot migrate from its current
         * location while the application is actively using it
         */
        hipMemAllocationTypePinned = 0x1,
        hipMemAllocationTypeMax = 0x7FFFFFFF
    } hipMemAllocationType;
    /**
     * @brief Flags for specifying handle types for memory pool allocations
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipMemAllocationHandleType
    {
        hipMemHandleTypeNone = 0x0,                ///< Does not allow any export mechanism
        hipMemHandleTypePosixFileDescriptor = 0x1, ///< Allows a file descriptor for exporting. Permitted only on POSIX systems
        hipMemHandleTypeWin32 = 0x2,               ///< Allows a Win32 NT handle for exporting. (HANDLE)
        hipMemHandleTypeWin32Kmt = 0x4             ///< Allows a Win32 KMT handle for exporting. (D3DKMT_HANDLE)
    } hipMemAllocationHandleType;
    /**
     * Specifies the properties of allocations made from the pool.
     */
    typedef struct hipMemPoolProps
    {
        hipMemAllocationType allocType;         ///< Allocation type. Currently must be specified as @p hipMemAllocationTypePinned
        hipMemAllocationHandleType handleTypes; ///< Handle types that will be supported by allocations from the pool
        hipMemLocation location;                ///< Location where allocations should reside
        /**
         * Windows-specific LPSECURITYATTRIBUTES required when @p hipMemHandleTypeWin32 is specified
         */
        void *win32SecurityAttributes;
        unsigned char reserved[64]; ///< Reserved for future use, must be 0
    } hipMemPoolProps;
    /**
     * Opaque data structure for exporting a pool allocation
     */
    typedef struct hipMemPoolPtrExportData
    {
        unsigned char reserved[64];
    } hipMemPoolPtrExportData;

    /*
     * @brief hipJitOption
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipJitOption
    {
        hipJitOptionMaxRegisters = 0,
        hipJitOptionThreadsPerBlock,
        hipJitOptionWallTime,
        hipJitOptionInfoLogBuffer,
        hipJitOptionInfoLogBufferSizeBytes,
        hipJitOptionErrorLogBuffer,
        hipJitOptionErrorLogBufferSizeBytes,
        hipJitOptionOptimizationLevel,
        hipJitOptionTargetFromContext,
        hipJitOptionTarget,
        hipJitOptionFallbackStrategy,
        hipJitOptionGenerateDebugInfo,
        hipJitOptionLogVerbose,
        hipJitOptionGenerateLineInfo,
        hipJitOptionCacheMode,
        hipJitOptionSm3xOpt,
        hipJitOptionFastCompile,
        hipJitOptionNumOptions
    } hipJitOption;
    /**
     * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
     */
    typedef enum hipFuncAttribute
    {
        hipFuncAttributeMaxDynamicSharedMemorySize = 8,
        hipFuncAttributePreferredSharedMemoryCarveout = 9,
        hipFuncAttributeMax
    } hipFuncAttribute;
    /**
     * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
     */
    typedef enum hipFuncCache_t
    {
        hipFuncCachePreferNone,   ///< no preference for shared memory or L1 (default)
        hipFuncCachePreferShared, ///< prefer larger shared memory and smaller L1 cache
        hipFuncCachePreferL1,     ///< prefer larger L1 cache and smaller shared memory
        hipFuncCachePreferEqual,  ///< prefer equal size L1 cache and shared memory
    } hipFuncCache_t;
    /**
     * @warning On AMD devices and some Nvidia devices, these hints and controls are ignored.
     */
    typedef enum hipSharedMemConfig
    {
        hipSharedMemBankSizeDefault,  ///< The compiler selects a device-specific value for the banking.
        hipSharedMemBankSizeFourByte, ///< Shared mem is banked at 4-bytes intervals and performs best
                                      ///< when adjacent threads access data 4 bytes apart.
        hipSharedMemBankSizeEightByte ///< Shared mem is banked at 8-byte intervals and performs best
                                      ///< when adjacent threads access data 4 bytes apart.
    } hipSharedMemConfig;
    /**
     * Struct for data in 3D
     *
     */
/*    typedef struct dim3
    {
        uint32_t x; ///< x
        uint32_t y; ///< y
        uint32_t z; ///< z
#ifdef __cplusplus
        constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};
#endif
    } dim3;
    */
    typedef struct hipLaunchParams_t
    {
        void *func;         ///< Device function symbol
        dim3 gridDim;       ///< Grid dimentions
        dim3 blockDim;      ///< Block dimentions
        void **args;        ///< Arguments
        size_t sharedMem;   ///< Shared memory
        hipStream_t stream; ///< Stream identifier
    } hipLaunchParams;
    typedef enum hipExternalMemoryHandleType_enum
    {
        hipExternalMemoryHandleTypeOpaqueFd = 1,
        hipExternalMemoryHandleTypeOpaqueWin32 = 2,
        hipExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
        hipExternalMemoryHandleTypeD3D12Heap = 4,
        hipExternalMemoryHandleTypeD3D12Resource = 5,
        hipExternalMemoryHandleTypeD3D11Resource = 6,
        hipExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    } hipExternalMemoryHandleType;
    typedef struct hipExternalMemoryHandleDesc_st
    {
        hipExternalMemoryHandleType type;
        union
        {
            int fd;
            struct
            {
                void *handle;
                const void *name;
            } win32;
        } handle;
        unsigned long long size;
        unsigned int flags;
    } hipExternalMemoryHandleDesc;
    typedef struct hipExternalMemoryBufferDesc_st
    {
        unsigned long long offset;
        unsigned long long size;
        unsigned int flags;
    } hipExternalMemoryBufferDesc;
    typedef void *hipExternalMemory_t;
    typedef enum hipExternalSemaphoreHandleType_enum
    {
        hipExternalSemaphoreHandleTypeOpaqueFd = 1,
        hipExternalSemaphoreHandleTypeOpaqueWin32 = 2,
        hipExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
        hipExternalSemaphoreHandleTypeD3D12Fence = 4
    } hipExternalSemaphoreHandleType;
    typedef struct hipExternalSemaphoreHandleDesc_st
    {
        hipExternalSemaphoreHandleType type;
        union
        {
            int fd;
            struct
            {
                void *handle;
                const void *name;
            } win32;
        } handle;
        unsigned int flags;
    } hipExternalSemaphoreHandleDesc;
    typedef void *hipExternalSemaphore_t;
    typedef struct hipExternalSemaphoreSignalParams_st
    {
        struct
        {
            struct
            {
                unsigned long long value;
            } fence;
            struct
            {
                unsigned long long key;
            } keyedMutex;
            unsigned int reserved[12];
        } params;
        unsigned int flags;
        unsigned int reserved[16];
    } hipExternalSemaphoreSignalParams;
    /**
     * External semaphore wait parameters, compatible with driver type
     */
    typedef struct hipExternalSemaphoreWaitParams_st
    {
        struct
        {
            struct
            {
                unsigned long long value;
            } fence;
            struct
            {
                unsigned long long key;
                unsigned int timeoutMs;
            } keyedMutex;
            unsigned int reserved[10];
        } params;
        unsigned int flags;
        unsigned int reserved[16];
    } hipExternalSemaphoreWaitParams;

#if __HIP_HAS_GET_PCH
    /**
     * Internal use only. This API may change in the future
     * Pre-Compiled header for online compilation
     *
     */
    void __hipGetPCH(const char **pch, unsigned int *size);
#endif

    /*
     * @brief HIP Devices used by current OpenGL Context.
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipGLDeviceList
    {
        hipGLDeviceListAll = 1,          ///< All hip devices used by current OpenGL context.
        hipGLDeviceListCurrentFrame = 2, ///< Hip devices used by current OpenGL context in current
                                         ///< frame
        hipGLDeviceListNextFrame = 3     ///< Hip devices used by current OpenGL context in next
                                         ///< frame.
    } hipGLDeviceList;

    /*
     * @brief HIP Access falgs for Interop resources.
     * @enum
     * @ingroup Enumerations
     */
    typedef enum hipGraphicsRegisterFlags
    {
        hipGraphicsRegisterFlagsNone = 0,
        hipGraphicsRegisterFlagsReadOnly = 1, ///< HIP will not write to this registered resource
        hipGraphicsRegisterFlagsWriteDiscard =
            2,                                        ///< HIP will only write and will not read from this registered resource
        hipGraphicsRegisterFlagsSurfaceLoadStore = 4, ///< HIP will bind this resource to a surface
        hipGraphicsRegisterFlagsTextureGather =
            8 ///< HIP will perform texture gather operations on this registered resource
    } hipGraphicsRegisterFlags;

    typedef struct _hipGraphicsResource hipGraphicsResource;

    typedef hipGraphicsResource *hipGraphicsResource_t;

    /**
     * An opaque value that represents a hip graph
     */
    typedef struct ihipGraph *hipGraph_t;
    /**
     * An opaque value that represents a hip graph node
     */
    typedef struct hipGraphNode *hipGraphNode_t;
    /**
     * An opaque value that represents a hip graph Exec
     */
    typedef struct hipGraphExec *hipGraphExec_t;

    /**
     * @brief hipGraphNodeType
     * @enum
     *
     */
    typedef enum hipGraphNodeType
    {
        hipGraphNodeTypeKernel = 1,            ///< GPU kernel node
        hipGraphNodeTypeMemcpy = 2,            ///< Memcpy 3D node
        hipGraphNodeTypeMemset = 3,            ///< Memset 1D node
        hipGraphNodeTypeHost = 4,              ///< Host (executable) node
        hipGraphNodeTypeGraph = 5,             ///< Node which executes an embedded graph
        hipGraphNodeTypeEmpty = 6,             ///< Empty (no-op) node
        hipGraphNodeTypeWaitEvent = 7,         ///< External event wait node
        hipGraphNodeTypeEventRecord = 8,       ///< External event record node
        hipGraphNodeTypeMemcpy1D = 9,          ///< Memcpy 1D node
        hipGraphNodeTypeMemcpyFromSymbol = 10, ///< MemcpyFromSymbol node
        hipGraphNodeTypeMemcpyToSymbol = 11,   ///< MemcpyToSymbol node
        hipGraphNodeTypeCount
    } hipGraphNodeType;

    typedef void (*hipHostFn_t)(void *userData);
    typedef struct hipHostNodeParams
    {
        hipHostFn_t fn;
        void *userData;
    } hipHostNodeParams;
    typedef struct hipKernelNodeParams
    {
        dim3 blockDim;
        void **extra;
        void *func;
        dim3 gridDim;
        void **kernelParams;
        unsigned int sharedMemBytes;
    } hipKernelNodeParams;
    typedef struct hipMemsetParams
    {
        void *dst;
        unsigned int elementSize;
        size_t height;
        size_t pitch;
        unsigned int value;
        size_t width;
    } hipMemsetParams;

    /**
     * @brief hipGraphExecUpdateResult
     * @enum
     *
     */
    typedef enum hipGraphExecUpdateResult
    {
        hipGraphExecUpdateSuccess = 0x0,              ///< The update succeeded
        hipGraphExecUpdateError = 0x1,                ///< The update failed for an unexpected reason which is described
                                                      ///< in the return value of the function
        hipGraphExecUpdateErrorTopologyChanged = 0x2, ///< The update failed because the topology changed
        hipGraphExecUpdateErrorNodeTypeChanged = 0x3, ///< The update failed because a node type changed
        hipGraphExecUpdateErrorFunctionChanged =
            0x4, ///< The update failed because the function of a kernel node changed
        hipGraphExecUpdateErrorParametersChanged =
            0x5, ///< The update failed because the parameters changed in a way that is not supported
        hipGraphExecUpdateErrorNotSupported =
            0x6, ///< The update failed because something about the node is not supported
        hipGraphExecUpdateErrorUnsupportedFunctionChange = 0x7
    } hipGraphExecUpdateResult;

    typedef enum hipStreamCaptureMode
    {
        hipStreamCaptureModeGlobal = 0,
        hipStreamCaptureModeThreadLocal,
        hipStreamCaptureModeRelaxed
    } hipStreamCaptureMode;
    typedef enum hipStreamCaptureStatus
    {
        hipStreamCaptureStatusNone = 0,   ///< Stream is not capturing
        hipStreamCaptureStatusActive,     ///< Stream is actively capturing
        hipStreamCaptureStatusInvalidated ///< Stream is part of a capture sequence that has been
                                          ///< invalidated, but not terminated
    } hipStreamCaptureStatus;

    typedef enum hipStreamUpdateCaptureDependenciesFlags
    {
        hipStreamAddCaptureDependencies = 0, ///< Add new nodes to the dependency set
        hipStreamSetCaptureDependencies,     ///< Replace the dependency set with the new nodes
    } hipStreamUpdateCaptureDependenciesFlags;

    typedef enum hipGraphInstantiateFlags
    {
        hipGraphInstantiateFlagAutoFreeOnLaunch =
            1, ///< Automatically free memory allocated in a graph before relaunching.
    } hipGraphInstantiateFlags;
#include <hip/amd_detail/amd_hip_runtime_pt_api.h>
