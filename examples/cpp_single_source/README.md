# C++ Single Source

These are examples of C++ programs which contain code to interact with a CPU-side API, and also one or more GPU kernels.

- the CPU-side API is used to do things like:
    - allocate GPU memory
    - copy data from main memory into GPU memory
    - launch a kernel on the GPU
    - copy data back from GPU memory into main memory
- the kernels will run on the GPU cores

## Prequisites

- have installed llvm 14, from https://apt.llvm.org/ or https://releases.llvm.org/download.html#14.0.0
- have python 3 in the PATH
- have cmake, g++, build-essential, e.g.
```
sudo apt-get install -y cmake g++ build-essential
```
- have built llc from branch D122918-zfinx-from-sunshaoce of https://github.com/hughperkins/llvm-project
```
git clone --filter=tree:0 https://github.com/hughperkins/llvm-project.git -b D122918-zfinx-from-sunshaoce
cd llvm-project
mkdir build
cd build
cmake -D EXPERIMENTAL_TARGETS_TO_BUILD:str=RISCV \
    -D LLVM_TARGETS_TO_BUILD:str=RISCV \
    -D CMAKE_BUILD_TYPE:STR=Release \
    -D LLVM_BUILD_RUNTIME:BOOL=OFF \
    -D LLVM_BUILD_RUNTIMES:BOOL=OFF \
    -D LLVM_BUILD_TOOLS:BOOL=OFF \
    -D LLVM_BUILD_UTILS:BOOL=OFF \
    -D LLVM_INCLUDE_BENCHMARKS:BOOL=OFF \
    -D LLVM_INCLUDE_DOCS:BOOL=OFF \
    -D LLVM_INCLUDE_EXAMPLES:BOOL=OFF \
    -D LLVM_INCLUDE_RUNTIMES:BOOL=OFF \
    -D LLVM_INCLUDE_TESTS:BOOL=OFF \
    -D LLVM_INCLUDE_UTILS:BOOL=OFF \
    ../llvm
make -j $(nproc) llc
sudo cp bin/llc /usr/local/bin/llc-zfinx
```
- have built the gpu simulator and runtime:
```
prot/verilator/prot_single_source/build-cmake.sh
```

## Running

To run:

```
examples/cpp_single_source/run.sh [example name]
```

e.g.

```
examples/cpp_single_source/run.sh matrix_mul
```
