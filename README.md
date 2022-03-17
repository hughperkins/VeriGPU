# toy_proc
Play with building a toy processor from scratch, in verilog

# Vision

Write a GPU, targeting ASIC tape-out. I don't actually intend to tape this out myself, but I intend to do what I can to verify somehow that tape-out would work ok, timings ok, etc.

Loosely compliant with RISC-V ISA. Where RISC-V conflicts with designing for a GPU setting, we break with RISC-V.

# Simulation

![toy proc workflow](https://raw.githubusercontent.com/hughperkins/toy_proc/main/img/toy_proc_workflow.png)

## prerequisites

- python 3
- iverilog (e.g. `brew install iverilog`, or see https://github.com/steveicarus/iverilog )

## procedure

```
python run.py --name [progname, e.g. prog5]
```

Under the hood, this will run `python assembler.py` to assemble the `.asm` file into a `.hex` byte-code file. Then, it will run `iverilog`, and execute the resulting object code. See [run.py](https://github.com/hughperkins/toy_proc/blob/main/run.py).

`progname` is the name of a `.asm` file in [examples](examples), e.g. `prog2` (without suffix).

Example output:

![Example output](https://raw.githubusercontent.com/hughperkins/toy_proc/main/img/example_output.png)

## Example assembler programs that run on this hardware

- sum integers from 0 to 5 [examples/prog16.asm](examples/prog16.asm)
- factorial of integers 0 to 5 [examples/prog18.asm](examples/prog18.asm)

# Testing

[![CircleCI](https://circleci.com/gh/hughperkins/toy_proc/tree/main.svg?style=svg)](https://circleci.com/gh/hughperkins/toy_proc/tree/main)

```
bash test/reg_test.sh
```

- under the hood, this will run many of the examples in [examples](examples), and check the outputs against the expected outputs, which are in the paired files, with suffix `_expected.txt`, also in [examples](examples) folder.

```
pytest -v
```
- run some unit tests on the assembler (which is in python)

## Testing using gate-level simulation

We use yosys to synthesize proc.sv to gate-level verilog netlist, then run this file using comp_driver.sv, as per
[test/reg_test.sh](test/reg_test.sh):

```
bash test/gls_proc.sh
```

# Timing

## Timing based on gate-level netlist


### Current result

You can see the current clock cycle propagation delay by opening the most recent build at [toy_proc circleci](https://app.circleci.com/pipelines/github/hughperkins/toy_proc?branch=main&filter=all), going to 'artifacts', and clicking on 'build/timing-proc.txt'. As of writing this, it was 110 nand gate units, i.e. equivalent to passing through about 110 nand units.
- at 90nm, one nand gate unit is about 50ps, giving a cycle time of about 5.5ns, and a frequency of about 200MHz
- at 5nm, one nand gate unit is about 5ps, giving a cycle time of about 0.55ns, and a frequency of about 2GHz
(Note: this analysis totally neglects layout, i.e. wire delay over distance, so it's just to give an idea).

### Concept

- we first use [yosys](http://bygone.clairexen.net/yosys/) to synthesize our verilog file to a gate-level netlist
    - a gate-level netlist is also a verilog file, but with the behavioral bits (`always`, etc.) removed, and operations such as `+`, `-` etc all replaced by calls to standard cells, such as `NOR2X1`, `NAND2X1`, etc
- then we use a custom script [toy_proc/timing.py](toy_proc/timing.py) to walk the graph of the resulting netlist, and find the longest propagation delay from the inputs to the outputs
    - the delay units are in `nand` propagation units, where a `nand` propagation unit is defined as the time to propagate through a single nand gate
    - a NOT gate is 0.6
    - an AND gate is 1.6 (it's a NAND followed by a NOT)
    - we assume that all cells only have a single output currently
- the cell propagation delays are loosely based on those in https://web.engr.oregonstate.edu/~traylor/ece474/reading/SAED_Cell_Lib_Rev1_4_20_1.pdf , which is a 90nm spec sheet, but could be representative of relative timings, which are likely architecture-independent
- you can see the relative cell times we use at [toy_proc/timing.py](https://github.com/hughperkins/toy_proc/blob/c4e37bdde601829f3959935e564503dbe30677fa/toy_proc/timing.py#L25-L46)

### Details

For more details see [docs/timings.md](docs/timing.md)

# Implementation details

## RISC-V instructions

```
SW    rs2, offset(rs1)     # use for both integers and floats
LW    rd,  offset(rs1)     # use for both integers and floats
ADDI  rd,  rs1, immediate
BEQ   rs1, rs2, location
BNE   rs1, rs2, location
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
location:  # to label a location that we will branch conditionally to
           # (for now, must precede the branch instruction)
```

## Pseudoinstructions

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
```

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

# Short-term plan

For long-term plan, see section Vision above. For short-term plan, see [todo.txt](docs/todo.txt). This also shows recent changes.
