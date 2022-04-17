# HIP

HIP API, as described in https://github.com/ROCm-Developer-Tools/HIP

## Test case for development

```
src/gpu_runtime/build-cmake.sh &&  python verigpu/test_hip.py
```

## Pre-requisites

- pytorch HIP only supported on linux
- must have installed pytorch HIP version
- must have gone into `lib/python*/site-packages/torch/lib`, and
    - removed/moved the following files:
        - libamdhip64.so
        - libhsa-runtime64.so
        - libroctx64.so
    - must have created symlinks from `${VERIGPU}/build-runtime/libverigpu_runtime.so`, where `${VERIGPU}` is the absolute path to the VeriGPU repository, to files in the above `lib` directory, with the exact same names as those moved
- must have all pre-requisites for running VeriGPU single-source, [examples/cpp_single_source/README.md](/examples/cpp_single_source/README.md)

## Results at time of writing

At time of writing, running this test-case gives:
```
hipInit
hipGetDeviceCount
hipGetDeviceCount
hipGetDeviceProperties
hipGetDeviceCount
hipGetDevice
hipGetDevice
hipGetDevice
hipGetDevice
hipGetLastError
hipMalloc size=2097152
gpuMalloc 2097152
 gpuMalloc candidate size 16777088 requested 2097152
hipSetDevice
hipSetDevice
hipGetDevice
hipGetDevice
hipMemcpyWithStream
hipGetLastError
hipGetErrorString
hipSetDevice
Traceback (most recent call last):
  File "/home/ubuntu/git/verigpu/verigpu/test_hip.py", line 4, in <module>
    a.cuda()
RuntimeError
```
