# Timing

## Concept

- we first use [yosys](http://bygone.clairexen.net/yosys/) to synthesize our verilog file to a gate-level netlist
    - a gate-level netlist is also a verilog file, but with the behavioral bits (`always`, etc.) removed, and operations such as `+`, `-` etc all replaced by calls to standard cells, such as `NOR2X1`, `NAND2X1`, etc
- then we use a custom script [toy_proc/timing.py](toy_proc/timing.py) to walk the graph of the resulting netlist, and find the longest propagation delay from the inputs to the outputs
    - the delay units are in `nand` propagation units, where a `nand` propagation unit is defined as the time to propagate through a single nand gate
    - a NOT gate is 0.6
    - an AND gate is 1.6 (it's a NAND followed by a NOT)
    - we assume that all cells only have a single output currently
- the cell propagation delays are loosely based on those in https://web.engr.oregonstate.edu/~traylor/ece474/reading/SAED_Cell_Lib_Rev1_4_20_1.pdf , which is a 90nm spec sheet, but could be representative of relative timings, which are likely architecture-independent
- you can see the relative cell times we use at [toy_proc/timing.py](https://github.com/hughperkins/toy_proc/blob/c4e37bdde601829f3959935e564503dbe30677fa/toy_proc/timing.py#L25-L46)

## Current result

You can see the current clock cycle propagation delay by opening the most recent build at https://app.circleci.com/pipelines/github/hughperkins/toy_proc, going to 'artifacts', and clicking on 'build/timing.txt'. As of writing this, it was 110 nand gate units, i.e. equivalent to passing through about 110 nand units.
- at 90nm, one nand gate unit is about 50ps, giving a cycle time of about 5.5ns, and a frequency of about 200MHz
- at 5nm, one nand gate unit is about 5ps, giving a cycle time of about 0.55ns, and a frequency of about 2GHz
(Note: this analysis totally neglects layout, i.e. wire delay over distance, so it's just to give an idea).

## Prerequities

- python3
- [yosys](http://bygone.clairexen.net/yosys/)
- have installed the following python packages
```
pip install networkx pydot
```

## Procedure

e.g. for the module at [prot/add_one_2chunks.sv](prot/add_one_2chunks.sv), run:

```
python toy_proc/timing.py --in-verilog prot/add_one_2chunks.sv
# optionally can use --cell-lib to specify path to cell library. By default will use osu018 cell library in `tech/osu018` folder
```

## Example outputs

```
# pure combinatorial models:
$ python toy_proc/timing.py --in-verilog prot/add_one.sv 
output max delay: 37.4 nand units
$ python toy_proc/timing.py --in-verilog prot/add_one_chunked.sv 
output max delay: 27.2 nand units
$ python toy_proc/timing.py --in-verilog prot/add_one_2chunks.sv 
output max delay: 24.6 nand units
$ python toy_proc/timing.py --in-verilog prot/mul.sv 
output max delay: 82.8 nand units
$ python toy_proc/timing.py --in-verilog prot/div.sv 
output max delay: 1215.8 nand units

# flip-flop modules:
$ python toy_proc/timing.py --in-verilog prot/clocked_counter.sv 
max propagation delay: 37.4 nand units

# the processor module itself :)
$ python toy_proc/timing.py --in-verilog src/proc.sv
max propagation delay: 101.6 nand units
```
