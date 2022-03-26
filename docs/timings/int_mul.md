# Timings for integer multiplication

Timings for float multiplication ultimately boil down to the time for an integer multiplication of the mantissas, and timings for integer mutliplication are ovviously already integer multiplications, so we only really need to consider integer multiplication.

We compare the area, the propagation delay, and the clock cycles, of various methods.

Area and propagation delay are both in 'nand units'. See [/docs/timing.md](/docs/timing.md) for more details.

## 24-bit

24-bit is for 32-bit float multiplication. The result is the full 48-bits, which, in the context of a float, is then right-shifted, and the exponent changed. This means the lower bits are often unused. Unfortunately, there's no obvious and general approach to calculate the upper bits without calculating also the lower bits, since the carry from summing the lower bits can modify the upper bits.

## 32-bit

32-bit is for direct use in 32-bit int multiplicaiton. In this case we can throw away the upper 32-bits of the result, meaning we don't even need to calculate them.

| Approach | Filepath | Delay (nand units) |Area (nand units) | Clock cycles |
|---------|-----------|---------------------|----------------|------------|
| `*` | [/prot/int/primitives/mul.sv](/prot/int/primitives/mul.sv) |  82.8 | 5370 | 1|
| naively sum partial products | [prot/int/mul/mul_partial_sum.sv](prot/int/mul/mul_partial_sum.sv) | 82.8 | 5372 | 1 |
| Dadda + chunk the final carry addition | [/toy_proc/generation/dadd.py](/toy_proc/generation/dadd.py) |  67.4 | 4960 | 1|
| Sum one partial product per clock cycle |  [prot/int/mul/mul_partial_per_cycle_task.sv](prot/int/mul/mul_partial_per_cycle_task.sv) | 97.8 | 2103 | 32 |
| Naively sum a single output bit per clock cycle | [prot/int/mul/mul_partial_add_task.sv](prot/int/mul/mul_partial_add_task.sv) | 82.8 |5372 | 32|
| Sum 1 output bit per clock cycle, using position-invariant adders | [prot/int/mul/mul_partial_add_invar_task.sv](prot/int/mul/mul_partial_add_invar_task.sv) | 69.8 | 908 | 32|
| Sum 2 output bits per clock cycle, using position-invariant adders | [prot/int/mul/mul_partial_add_invar_task.sv](prot/int/mul/mul_partial_add_invar_task.sv) |  75.6|1414|16|
| Sum 4 output bits per clock cycle, using position-invariant adders | [prot/int/mul/mul_partial_add_invar_task.sv](prot/int/mul/mul_partial_add_invar_task.sv) |  86.6|1970 | 8|
| Sum 8 output bits per clock cycle, using position-invariant adders | [prot/int/mul/mul_partial_add_invar_task.sv](prot/int/mul/mul_partial_add_invar_task.sv) | 
| Sum 1 output bit per clock cycle, using Dadda/Booth-style adders | [toy_proc/generation/mul_pipeline_cycle.py](toy_proc/generation/mul_pipeline_cycle.py) | 58.2 | 863 | 32 |
| Sum 2 output bit per clock cycle, using Dadda/Booth-style adders | [toy_proc/generation/mul_pipeline_cycle.py](toy_proc/generation/mul_pipeline_cycle.py) | 63.2 | 1244 | 16 |
| Sum 4 output bit per clock cycle, using Dadda/Booth-style adders | [toy_proc/generation/mul_pipeline_cycle.py](toy_proc/generation/mul_pipeline_cycle.py) | 74.8 | 2018 | 8 |
| Sum 8 output bit per clock cycle, using Dadda/Booth-style adders | [toy_proc/generation/mul_pipeline_cycle.py](toy_proc/generation/mul_pipeline_cycle.py) | 93.2 | 3429 | 4|
