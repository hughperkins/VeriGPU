# src folder

- gpu_card.sv represents the GPU card, including gpu_die.sv, and simulated off-chip global memory
- gpu_die.sv represents the GPU die
- core.sv represensts a single GPU core
- gpu_controller.sv handles communications with the main cpu, and main memory

The sub-folder [gpu_runtime](gpu_runtime) contains C++ source-code for a C++ runtime library, that handles:
- compilation of C++ single source programs into device side and hostside code
- communications with gpu_controller.sv, in order to send data to and from the GPU global memory, and to launch kernels
- GPU memory allocation (handled host-side)
