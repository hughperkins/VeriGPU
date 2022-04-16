# GPU runtime

This is C++ that runs on the host-side, i.e. on the CPU, and handles things like:
- copying data to the GPU, via the GPU Controller
- receiving data from the GPU, again via the GPU Controller
- allocating data on the GPU
- sending kernels to the GPU, via the GPU controller
- launching kernels on the GPU, via the GPU controller

# To build

## Pre-requisites

- have installed llvm 13 or 14, e.g. from https://apt.llvm.org/ or https://releases.llvm.org/download.html#14.0.0
- have installed verilator
- ccmake
- g++
- expect (for `unbuffer`)
- build-essential (includes make and similar build tools)

## Procedure

```
bash src/gpu_runtime/build-cmake.sh
```
- fix any issues in src/gpu_rutnime/CMakeLists.txt and/or src/gpu_runtime/build-cmake.sh
- if any changes to either of these files, consider creating a PR for the changes, and/or creating an issue on github.
