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

# To run self-tests

```
bash src/reg_test.sh
```

- under the hood, this will run many of the examples in [examples](examples), and check the outputs against the expected outputs, which are in the paired files, with suffix `_expected.txt`, also in [examples](examples) folder.

# To compile with verilator

(there is no runner for verilator currently, but the modules do compile)

## prerequisites

- verilator (e.g. `brew install verilator`)

## procedure

```
verilator -sv --cc src/proc.sv src/mem.sv src/comp.sv -Isrc
```

# To determine maximum clockspeed / timing

[This section in progress]

## prequisites

- yosys (e.g. `brew install yosys`, or see http://bygone.clairexen.net/yosys/ )
- opensroad/sta (build from source, see https://github.com/The-OpenROAD-Project/OpenSTA )

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

No caching of any sort is implemented currently (no level1, no level2, no level2, not even instruction cache :P ). Since I intend to target creating a GPU, which has a different cache mechanism than CPU, I'll think about this once it starts to look more like a GPU.

## I/O

- any word written to location 1000 will be considered to have been sent to a memory-mapped i/o device, which will write this value out, in our case to stdout, via the test bench code.
- writing any word to location 1004 halts the simulation.

# Short-term plan

For long-term plan, see section Vision above. For short-term plan, see [todo.txt](docs/todo.txt)
