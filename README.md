# toy_proc
Play with building a toy processor from scratch, in verilog

# Vision

Experiment with writing some very simple toy processor, maybe GPU, targeting ASIC synthesis. I don't actually intend to run synthesis myself, but I intend to do what I can to verify somehow that synthesis would work ok, timings ok, etc.

Loosely compliant with RISC-V ISA.

# Simulation Workflow

![toy proc workflow](https://raw.githubusercontent.com/hughperkins/toy_proc/main/img/toy_proc_workflow.png)

# To run / simulate

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

# Some example programs that run ok

- sum integers from 0 to 5 [examples/prog16.asm](examples/prog16.asm)

# To run self-tests

```
bash src/reg_test.sh
```

- under the hood, this will run many of the examples in [examples](examples), and check the outputs against the expected outputs, which are in the paired files, with suffix `_expected.txt`, also in [examples](examples) folder.

# To compile with verilator

(there is no runner for verilator currently, but I do test sometimes that the modules do compile; verilator is great for catching mismatches between bus sizes, and some other errors).

## prerequisites

- verilator (e.g. `brew install verilator`)

## procedure

```
verilator -sv --cc src/proc.sv src/mem.sv src/comp.sv -Isrc
```

# Timing

## Timing based on gate-level netlist

### Concept

- we first use [yosys](http://bygone.clairexen.net/yosys/) to synthesize our verilog file to a gate-level netlist
    - a gate-level netlist is also a verilog file, but with the behavioral bits (`always`, etc.) removed, and operations such as `+`, `-` etc all replaced by calls to standard cells, such as `NOR2X1`, `NAND2X1`, etc
- then we use a custom script [src/timing.py](src/timing.py) to walk the graph of the resulting netlist, and find the longest propagation delay from the inputs to the outputs
    - the delay units are in `nand` propagation units, where a `nand` propagation unit is defined as the time to propagate through a single nand gate
    - a NOT gate is 0.6
    - an AND gate is 1.6 (it's a NAND followed by a NOT)
    - we assume that all cells only have a single output currently
- the cell propagation delays are loosely based on those in https://web.engr.oregonstate.edu/~traylor/ece474/reading/SAED_Cell_Lib_Rev1_4_20_1.pdf , which is a 90nm spec sheet, but could be representative of relative timings, which are likely architecture-independent

### Constraints

- currently only works on fully combinatorial modules

### Prerequities

- python3
- [yosys](http://bygone.clairexen.net/yosys/)
- have installed the following packages
```
pip install networkx 
```

### Procedure

e.g. for the module at [prot/add_one_2chunks.sv](prot/add_one_2chunks.sv), run:

```
python src/timing.py --in-verilog prot/add_one_2chunks.sv
```

### Example outputs

```
$ python src/timing.py --in-verilog prot/add_one.sv 
output max delay: 37.4 nand units
$ python src/timing.py --in-verilog prot/add_one_chunked.sv 
output max delay: 27.2 nand units
$ python src/timing.py --in-verilog prot/add_one_2chunks.sv 
output max delay: 24.6 nand units
$ python src/timing.py --in-verilog prot/mul.sv 
output max delay: 82.8 nand units
$ python src/timing.py --in-verilog prot/div.sv 
output max delay: 1215.8 nand units
```

## Timing after running full layout

[This section in progress]

### Prequisites

- yosys (e.g. `brew install yosys`, or see [http://bygone.clairexen.net/yosys/](http://bygone.clairexen.net/yosys/))
- opensroad/sta (build from source, see https://github.com/The-OpenROAD-Project/OpenSTA )

### Procedure

```
yosys -s src/yosys.tacl
# some sta command here that I haven't figure out yet :)
```

# Technical details

## RISC-V instructions

```
SW rs2, offset(rs1)
LW rd, offset(rs1)
ADDI rd, rs1, immediate
BEQ  rs1, rs2, location
BNE  rs1, rs2, location
ADD  rd, rs1, rs2
SUB  rd, rs1, rs2
SLL  rd, rs1, rs2
SRL  rd, rs1, rs2
AND  rd, rs1, rs2
OR   rd, rs1, rs2
XOR  rd, rs1, rs2
SLTU rd, rs1, rs2
location:  # to label a location that we will branch conditionally to
           # (for now, must precede the branch instruction)
```

## Pseudoinstructions

```
LI rd, immediate  # loads immediate into register rd
HALT              # halts simulation
OUT immediate     # sends immediate to stdout, via writing to mem location 1000
OUTR rd           # sends contents of register rd to stdout
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

## I/O

- any word written to location 1000 will be considered to have been sent to a memory-mapped i/o device, which will write this value out, in our case to stdout, via the test bench code.
- writing any word to location 1004 halts the simulation.

# Short-term plan

For long-term plan, see section Vision above. For short-term plan, see [todo.txt](docs/todo.txt)

# Recent changes

- created script timing.py, that measures longest propagation time, for combinatorial modules, based on gate-level netlist
