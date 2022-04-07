# OpenSource GPU

Build an opensource GPU, targeting ASIC tape-out, for [machine learning](https://en.wikipedia.org/wiki/Machine_learning)  ("ML"). Hopefully, can get it to work with the [PyTorch](https://pytorch.org) deep learning framework.

# Vision

Create an opensource GPU for machine learning.

I don't actually intend to tape this out myself, but I intend to do what I can to verify somehow that tape-out would work ok, timings ok, etc.

Intend to implement a [HIP](https://github.com/ROCm-Developer-Tools/HIP) API, that is compatible with [pytorch](https://pytorch.org) machine learning framework. Open to provision of other APIs, such as [SYCL](https://www.khronos.org/sycl/) or [NVIDIA® CUDA™](https://developer.nvidia.com/cuda-toolkit).

Internal GPU Core ISA loosely compliant with [RISC-V](https://riscv.org/technical/specifications/) ISA. Where RISC-V conflicts with designing for a GPU setting, we break with RISC-V.

Intend to keep the cores very focused on ML. For example, [brain floating point](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) ("BF16") throughout, to keep core die area low. This should keep the per-core cost low. Similarly, Intend to implement only few float operations critical to ML, such as `exp`, `log`, `tanh`, `sqrt`.

# Architecture

End-to-end Architecture, including Host-side compilation and runtime:

![End-to-end Architecture](/docs/img/architecture.png)

GPU Die Architecture:

![GPU Die Architecture](/docs/img/GPU_die_architecture.png)

# Simulation

<!-- ![toy proc workflow](/docs/img/toy_proc_workflow.png) -->

![Example output](/docs/img/example_output.png)

# Planning

What direction are we thinking of going in? What works already? See:

- [docs/planning.md](docs/planning.md)

# Tech details

Our assembly language implementation and progress. Design of GPU memory, registers, and so on. See:

- [docs/tech_details.md](docs/tech_details.md)

# Verification

If we want to tape-out, we need solid verification. Read more at:

- [docs/verification.md](docs/verification.md)

# Metrics

we want the GPU to run quickly, and to use minimal die area. Read how we measure timings and area at:

- [docs/metrics.md](docs/metrics.md)
