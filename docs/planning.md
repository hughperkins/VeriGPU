# Planning

## Overall plan

We aim to design an opensource GPU, ready for tape-out, that can run machine learning training, using [HIP](https://github.com/ROCm-Developer-Tools/HIP) (for compatibility purposes). There are some steps to go through to get there :)

System components

|Component                     |Status|
|------------------------------|------|
|GPU controller          |Exists|
|Parallel instruction execution|Planned|
| Out of order execution       |Won't do (see below)|
| Data memory caching           |Planned|
| Instruction memory cachine   |Planned|
| Integer APU                  |Exists, for + / - +|
| Floating point unit          |Exists, for * +|
| Branching                    |Done|
| Single instruction multiple thread (SIMT) |Planned|
|Assembler                      |Exists|
| Single-source C++ compilation              |Exists|
| HIP compiler              |Planned|
| SYCL compiler              |Maybe|
| CUDA® compiler (for compatibility} |Maybe|

## Design decisions

We do not currently intend to implement out of order execution, meaning starting an instruction before the previous one has started, because this is complicated in a single-instruction multi-threading (SIMT) scenario, and because this makes the cores use more die area, and therefore fewer in number (or more expensive).

We will on the other hand implement parallel instruction execution, where we start an instruction whilst the previous instruction is still running. This is standard and fairly light-weight, doesn't take up too much die area.

The GPU is aimed squarely at machine learning training. Therefore it should ideally be compatible with current machine learning frameworks, such as [PyTorch](https://pytorch.org) and [Tensorflow](https://www.tensorflow.org/). This means that it almost certainly needs to be compatible with [NVIDIA® CUDA™](https://developer.nvidia.com/cuda-toolkit) or [AMD HIP](https://github.com/ROCm-Developer-Tools/HIP). However, we might also implement a OpenCL™ or SYCL interface, though support by major frameworks is currently limited. There is a dedicated OpenCL deep learning framework at [DeepCL](https://github.com/hughperkins/DeepCL), but it has a relatively limited set of neural network layers, and possible network topologies, compared to PyTorch and Tensorflow. There is a port of the old lua torch to OpenCL at [https://github.com/hughperkins/cltorch](https://github.com/hughperkins/cltorch), however the vast majority of ML practioners have now moved onto PyTorch, or use Tensorflow.

## Tactical day-to-day backlog

[todo.txt](todo.txt)
