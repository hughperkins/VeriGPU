# toy_proc

Build an opensource GPU, in verilog. Hopefully we can get it to run either [OpenCL™](https://www.khronos.org/opencl/) or [CUDA®](https://developer.nvidia.com/cuda-zone) (for compatibility purposes), or both, but it's early days for now :)

# Vision

Write a GPU, targeting ASIC tape-out. I don't actually intend to tape this out myself, but I intend to do what I can to verify somehow that tape-out would work ok, timings ok, etc.

Loosely compliant with RISC-V ISA. Where RISC-V conflicts with designing for a GPU setting, we break with RISC-V.

# Simulation

![toy proc workflow](https://raw.githubusercontent.com/hughperkins/toy_proc/main/img/toy_proc_workflow.png)

![Example output](https://raw.githubusercontent.com/hughperkins/toy_proc/main/img/example_output.png)

# Planning

What direction are we thinking of going in? What works already? See:

- [docs/planning.md](docs/planning.md)

# Tech details

Our assembly language implementation and progress. Design of GPU memory, registers, and so on. See:

- [docs/tech_details.md](docs/tech_details.md)

# Verification

If we want to tape-out, we need solid verification. Read more at:

- [docs/verification.md](docs/verification.md)

# Timings

we want the GPU to run quickly. Read how we measure timings at:

- [docs/timing.md](docs/timing.md)
