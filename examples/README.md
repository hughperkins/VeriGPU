# Examples

- [direct](direct): here we have assembler code that can be run directly on a single GPU core, as though it was running on a CPU
    - these assembly codes simply start running, without jumping, or pushing parameters onto a stack, and run until they reach `HALT`
    - these can be run by using `python run.py --name [example name]`
- [cpp_single_source](cpp_single_source): c++ code, that contains a GPU kernel, that we will compile, and run
    - the source-code contains both host-side code, that does things like allocate memory on the gpu, copy in data, from system main-memory, and launch a kernel
    - and it contains at least one GPU kernel, which will be compiled, pushed into the GPU, and then run

(Note that the easiest way to run these might be to use the Docker image [hughperkins/chip_design:latest](https://hub.docker.com/r/hughperkins/chip_design), since this is what is used to run [CI](https://app.circleci.com/pipelines/github/hughperkins/VeriGPU?branch=main&filter=all), so if the CI is green light, you know that this container works ok :) )
