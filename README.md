# toy_proc
Play with building a toy processor from scratch, in verilog

# Vision

Experiment with writing some very simple toy processor, maybe GPU, targeting ASIC synthesis. I don't actually intend to run synthesis myself, but I intend to do what I can to verify somehow that synthesis would work ok, timings ok, etc.

Loosely based to be compliant with RISC-V ISA (but some initial scaffolding instructions, and where being RV compliant would increase the workload a lot, starting with simpler instructions (e.g. implementing a loosely implemented `LI` for now, rather than `LUI` with `ADDI`).

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

# Recent updates

- migrated instruction layout to be compatible RISC-V
- migrated data and address buses to 32-bit, to be compatible with RV32
- created regression tests
