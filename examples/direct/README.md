# Direct examples

These examples are assembler code that can be run directly on a single core, as though that core was a CPU.

They are written primarily in RISCV-32 assembler. There are some exotic pseudoinstructions used such as `OUTR`, `OUTR.S` and `HALT`.

Exotic pseudoinstructions used:
- `OUTR`: sends the value of an unsigned integer from a register to some kind of console (via writing to memory location 1000)
- `OUTR.S`, as for `OUTR`, but treats the register as containing a 32-bit float
- `HALT`: halts the core


You can run these examples by using `python run.py [name]`, e.g.:

```
python run.py sum_ints
```
