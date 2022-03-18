# Assembler

In order to run the processor, we need to provide a program for the processor to run. The verilog code in this repo describes the hardware, but we also need to provide software to run on this hardware.

We use a python script [toy_proc/assembler.py](toy_proc/assembler.py) to convert assembly code into binary code that we can load into the processor simulations.

## Allowed isntructions

See [docs/tech_details.md](docs/tech_details.md)

## Example assembly programs

See [examples](examples)

Some specific programs:

- sum integers from 0 to 5 [examples/prog16.asm](examples/prog16.asm)
- factorial of integers 0 to 5 [examples/prog18.asm](examples/prog18.asm)