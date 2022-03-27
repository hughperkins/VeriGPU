# Implementation details

![verigpu workflow](https://raw.githubusercontent.com/hughperkins/verigpu/main/docs/img/toy_proc_workflow.png)

## Processor instruction set

We are currently loosely basing the processor instruction set on the [RISC-V ISA](https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf). The following sections list the available instructions in the [assembler](/verigpu/assembler.py).

### RISC-V instructions

```
SW    rs2, offset(rs1)     # use for both integers and floats
LW    rd,  offset(rs1)     # use for both integers and floats
ADDI  rd,  rs1, immediate
SLTIU rd,  rs1, immediate
SLLI  rd,  rs1, immediate
SRLI  rd,  rs1, immediate
ANDI  rd,  rs1, immediate
ORI  rd,  rs1, immediate
XORI  rd,  rs1, immediate
BEQ   rs1, rs2, location
BNE   rs1, rs2, location
BLTU   rs1, rs2, location
BGEU   rs1, rs2, location
ADD   rd,  rs1, rs2
SUB   rd,  rs1, rs2
SLL   rd,  rs1, rs2
SRL   rd,  rs1, rs2
AND   rd,  rs1, rs2
OR    rd,  rs1, rs2
XOR   rd,  rs1, rs2
SLTU  rd,  rs1, rs2
MUL   rs,  rs1, rs2
LI    rd   immediate
AUIPC rd   immediate
DIVU  rd,  rs1, rs2
MODU  rd,  rs1, rs2
location:  # to label a location that we can branch conditionally to
```

### Pseudoinstructions

```
LI rd, immediate  # loads immediate into register rd (works with floats, hex, binary, decimal)
HALT              # halts simulation
OUT immediate     # sends immediate to stdout as int, via writing to mem location 1000
OUTR rd           # sends contents of register rd to stdout, as int
OUTR.S rd           # sends contents of register rd to stdout, as float, via mem location 1008
OUTLOC immediate  # sends contents of memory location at immediate to stdout
NOP
MV rd, rs
NEG rd, rs
BEQZ rs, offset
BNEZ rs, offset
BLTZ rs, offset
BGTZ rs, offset
BLEZ rs, offset
BGEZ rs, offset
BLEU rs1, rs2, offset
BGTU rs1, rs2, offset
```

## Assembler

In order to run the processor, we need to provide a program for the processor to run. The verilog code in this repo describes the hardware, but we also need to provide software to run on this hardware.

We use a python script [verigpu/assembler.py](/verigpu/assembler.py) to convert assembly code into binary code that we can load into the processor simulations.

### Example assembly programs

See [examples](/examples)

Some specific programs:

- sum integers from 0 to 5 [examples/sum_ints.asm](/examples/sum_ints.asm)
- factorial of integers 0 to 5 [examples/calc_factorial.asm](/examples/calc_factorial.asm)
- prime numbers up to 31 [examples/calc_primes.asm](/examples/calc_primes.asm)
- sieve of eratosthenes, up to 31 [examples/sieve_eratosthenes.asm](/examples/sieve_eratosthenes.asm)

## Memory

Memory access is via a mock memory controller, which will wait several cycles before returning or writing data. A single word at a time can be read or written currently. Protocol is:

- set `addr` to read/write address
- if writing, set `wr_data` to data to write
- set `wr_req` or `rd_req` to 1, according to writing or reading, respectively
- (wait one cycle)
- set `wr_req` and `rd_req` to 0
- (wait cycles for `ack` to be set to 1)
- then, if reading, read the value from `rd_data`
- in either case, can immediately submit new request, on same clock cycle

No caching of any sort is implemented currently (no level1, no level2, no level3, not even instruction cache :P ). Since I intend to target creating a GPU, which has a different cache mechanism than CPU, I'll think about this once it starts to look more like a GPU.

## Registers

There are 31 registers, x1 to x31, along with x0, which is always 0s. Use the same registers for both integers and floats. (this latter point deviates from RISC-V, because we are targeting creating a GPU, where locality is based around each of thousands of tiny cores, rather than around the FP unit vs the integer APU).

## I/O

- any word written to location 1000 will be considered to have been sent to a memory-mapped i/o device, which will write this value out, in our case to stdout, via the test bench code.
- writing any word to location 1004 halts the simulation.
- writing a word to location 1008 outputs it as as a 32-bit float
