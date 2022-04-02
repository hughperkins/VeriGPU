# prot1

This prototype is to learn how to use vpi, to pass values in both directions.

hello.sv is the main module, that is started initially. It can then call into a vpi library created from hello.cpp.

## Setup/installation

- you need icarus verilog installed

## To run

```
prot/vpi/prot1/run.sh
```

## Expected output

```
$ prot/vpi/prot1/run.sh
++ dirname prot/vpi/prot1/run.sh
+ scriptdir=prot/vpi/prot1
+ echo scriptdir prot/vpi/prot1
scriptdir prot/vpi/prot1
+ cd prot/vpi/prot1
+ [[ ! -e build ]]
+ cd build
+ iverilog-vpi ../hello.cpp
Compiling ../hello.cpp...
Making hello.vpi from  hello.cpp.o...
+ iverilog -g2012 -ohello.vvp ../hello.sv
../hello.sv:21: warning: @* found no sensitivities so it will never trigger.
+ vvp -M. -mhello hello.vvp
hello, world
23
cpp_out 4294967173
i 0 x
i 1 222
i 2 333
i 3 444
i 4 x
i 5 555
i 6 x
i 7 x
i 8 x
i 9 x
i 10 x
i 11 x
i 12 x
i 13 x
i 14 x
i 15 x
```
