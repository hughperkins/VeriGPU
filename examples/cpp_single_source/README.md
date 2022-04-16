# C++ Single Source

These are examples of C++ programs which contain code to interact with a CPU-side API, and also one or more GPU kernels.

- the CPU-side API is used to do things like:
    - allocate GPU memory
    - copy data from main memory into GPU memory
    - launch a kernel on the GPU
    - copy data back from GPU memory into main memory
- the kernels will run on the GPU cores

## Prequisites

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
