# Installation

- install a recent version of [python3](https://www.python.org/)   (Used for the assembler, and to run the various simulations, syntheses, and verification tests)
- install [yosys](https://yosyshq.net/yosys/)  (Used for synthesis)
- install [iverilog](https://steveicarus.github.io/iverilog/)  (Used for simulation)
- install [verilator](https://www.veripool.org/verilator/)  (Used as an additional linter)
- install python libraries:
```
pip install -e .
pip install -r test/py/requirements.txt
```
