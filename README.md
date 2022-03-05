# toy_proc
Play with building a toy processor from scratch, in verilog

# To run / simulate

## prerequisites

- python 3
- iverilog (e.g. `brew install iverilog`, or see https://github.com/steveicarus/iverilog )

## procedure

```
python run.py --name [progname, e.g. prog5]
```

Under the hood, this will run `iverilog`, and execute it, see [run.py](https://github.com/hughperkins/toy_proc/blob/main/run.py).

`progname` is the name of a `.asm` file in `examples`, e.g. `prog2` (without suffix).

# To run self-tests

```
bash src/reg_test.sh
```

- under the hood, this will run many of the examples in [examples](examples), and check the outputs against the expected outputs.

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
