# Things that used to be in readme.md

Moved them here to keep readme.md clean(er)

# To compile with verilator

(there is no runner for verilator currently, but I do test sometimes that the modules do compile; verilator is great for catching mismatches between bus sizes, and some other errors).

## prerequisites

- verilator (e.g. `brew install verilator`)

## procedure

```
verilator -sv --cc src/proc.sv src/mem.sv src/comp.sv -Isrc
```


## Timing after running full layout

[This section in progress]

### Prequisites

- yosys (e.g. `brew install yosys`, or see [http://bygone.clairexen.net/yosys/](https://yosyshq.net/yosys/))
- opensroad/sta (build from source, see https://github.com/The-OpenROAD-Project/OpenSTA )

### Procedure

```
yosys -s src/yosys.tacl
# some sta command here that I haven't figure out yet :)
```

