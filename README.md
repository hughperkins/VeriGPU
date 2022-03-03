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

# To determine maximum clockspeed / timing

[This section in progress]

## prequisites

- yosys (e.g. `brew install yosys`, or see http://bygone.clairexen.net/yosys/ )
- opensroad/sta (build from source, see https://github.com/The-OpenROAD-Project/OpenSTA )

```
yosys -s src/yosys.tacl
# some sta command here that I haven't figure out yet :)
```
