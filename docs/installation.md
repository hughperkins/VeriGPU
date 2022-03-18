# Installation

- install a recent version of [python3](https://www.python.org/)   (Used for the assembler, and to run the various simulations, syntheses, and verification tests)
- install [yosys](http://bygone.clairexen.net/yosys/)  (Used for synthesis)
- install [iverilog](http://iverilog.icarus.com/)  (Used for simulation)
- install [verilator](https://www.veripool.org/verilator/)  (Used as an additional linter)
- install python libraries:
```
pip install -e .
pip install -r test/requirements.txt
```
