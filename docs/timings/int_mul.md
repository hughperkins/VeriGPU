# Timings for integer multiplication

Timings for float multiplication ultimately boil down to the time for an integer multiplication of the mantissas, and timings for integer mutliplication are ovviously already integer multiplications, so we only really need to consider integer multiplication.

We compare the area, the propagation delay, and the clock cycles, of various methods.

Area and propagation delay are both in 'nand units'. See [/docs/timing.md](/docs/timing.md) for more details.

## Concepts

In order to multiply two binary numbers, `a` and `b`, we form partial products, and add them together. In the case of binary numbers, the partial products consist of `a`, over and over again, but shifted. We look at `b`: for each `1` bit in `b`, we add another row for `a`, with `a` shifted by a number of bits corresponding to the position of the `1` in the `b`. For example, let's say we multiply `a` = `0b011` by `b` = `0b101`. We know this is 3 x 5, and the answer is 15, `0b1111`. To obtain the answer in binary, we will have `a` twice, one for each of the `1`s in `b`:

```
   011   b[0] = 1
 011     b[2] = 2
```
Amd then we add these together:
```
   011
+011
=01111
```

This works for all unsigned binary integers.

We can implement this directly in verilog in a couple of ways:
- simply write `a * b`. yosys, or some other synthesizer will convert this into a sum of partial products, as above
- write out the sum of partial products ourselves, in an `always @(*)` block

However, both of these approaches are purely combinatorial, so will run in a single clock cycle. This might push up our maximum delay propagation metric, reducing the maximum possible clock speed of the device. In addition, the carry bit needs to propagate all the way from the right-most output bit to the leftmost possible output bit. In the case of 32-bit integer multiplication, where the output is truncated to 32-bits, this creates a carry chain of 32-bits, leading to long propagation delay. In the case of float multiplication, which ultimately leads to integer multiplication of 24-bit mantissas, the output has 48-bits, which gives an even longer carrry chain. Finally, since we are summing a number of partial sums equal to the output width, there will be additional propagation delay vertically, through this stack of partial sums to add.

In order to reduce the carry chain length, there are a few things we can do:
- full adders at each position of the sum of partial sums, in order to reduce the bits, without creating propagation delay vertically, down the stack of partial products to sum. Booth and Dadda are two examples of this, https://en.wikipedia.org/wiki/Booth%27s_multiplication_algorithm and https://en.wikipedia.org/wiki/Dadda_multiplier
- when using a Booth or Dadda multiplier or similar, chunk the final carried sum at the end. For example, we could divide into two chunks. Although we don't know the carry from the right-most chunk until we have finished adding that chunk, we can calculate the left most chunk twice in the meantime: once for a carry of 0, and once for a carry of 1. Then, once we have the result from the right-most chunk, we can multiplex between them. This can reduce the carry-chain propagation delay by around half.
- we can split up the multiplication across multiple clock cycles. This will add clock cycle latency, but it will keep the per cycle propagation delay down, allowing us to run the cpu clock at high speed.

### Dadda multipliers

Booth, Dadda etc can be found in wikipedia, as above. We implement Dadda by writing a generator, in python, [/verigpu/generation/dadda.py](/verigpu/generation/dadda.py). The generated files are stored in git at [/src/generated](/src/generated).

### Splitting across clock cycles

There are a number of ways we can try to split across clock cycles:

- just add a few partial sums each clock cycle. In the limit, we can simply add a single one each time. This is implemented in 'Sum one partial product per clock cycle' below.
- we could imagine that we could implement part of the Dadda tree each clock cycle, and save the result so far in registers for the next clock cycle. Unfortuanately this would be prohibitively expensive in registers. For 24-bits, there are 504 gates in the correspondign Dadda tree. If we divide that into clock cycles, we will need to store a lot of state each clock cycle
- keep the inputs to each time step as the original `a` and `b`, and just calculate a few bits of the output each timestep. Pass a small `carry` vector, or 5 bits or so, between time-steps. There is a limit to how short each time-step can be using this approach, because we have 5 bits of carry for 24 or 32 bit multiplication. However, each time-step is identical, and relatively compact, so the die area used is small. The propagation delay is relatively short. This approach is listed as 'Sum x output bit per clock cycle, using xxx approach' below.
    - When we calculate the output bits each clock cycle, we can simply naively use standard adders to sum across the partial products, which is approach 'using position-invariant adders' below
    - Alternatively we can use full-adders, similar to how Booth or Dadda works, to reduce three bits by each adder at a time, without introducing carry chains until the end. This is the approach 'using Dadda/Booth-style adders' below.

## 24-bit

24-bit is for 32-bit float multiplication. The result is the full 48-bits, which, in the context of a float, is then right-shifted, and the exponent changed. This means the lower bits are often unused. Unfortunately, there's no obvious and general approach to calculate the upper bits without calculating also the lower bits, since the carry from summing the lower bits can modify the upper bits.

## 32-bit

32-bit is for direct use in 32-bit int multiplicaiton. In this case we can throw away the upper 32-bits of the result, meaning we don't even need to calculate them.

| Approach | Filepath | Delay (nand units) |Area (nand units) | Clock cycles |
|---------|-----------|---------------------|----------------|------------|
| `*` | [/prot/int/primitives/mul.sv](/prot/int/primitives/mul.sv) |  82.8 | 5370 | 1|
| naively sum partial products | [prot/int/mul/mul_partial_sum.sv](prot/int/mul/mul_partial_sum.sv) | 82.8 | 5372 | 1 |
| Dadda + chunk the final carry addition | [/verigpu/generation/dadda.py](/verigpu/generation/dadda.py) |  67.4 | 4960 | 1|
| Sum one partial product per clock cycle |  [/prot/int/mul/mul_partial_per_cycle_task.sv](/prot/int/mul/mul_partial_per_cycle_task.sv) | 97.8 | 2103 | 32 |
| Naively sum a single output bit per clock cycle | [/prot/int/mul/mul_partial_add_task.sv](/prot/int/mul/mul_partial_add_task.sv) | 82.8 |5372 | 32|
| Sum 1 output bit per clock cycle, using position-invariant adders | [/prot/int/mul/mul_partial_add_invar_task.sv](/prot/int/mul/mul_partial_add_invar_task.sv) | 69.8 | 908 | 32|
| Sum 2 output bits per clock cycle, using position-invariant adders | [/prot/int/mul/mul_partial_add_invar_task.sv](/prot/int/mul/mul_partial_add_invar_task.sv) |  75.6|1414|16|
| Sum 4 output bits per clock cycle, using position-invariant adders | [/prot/int/mul/mul_partial_add_invar_task.sv](/prot/int/mul/mul_partial_add_invar_task.sv) |  86.6|1970 | 8|
| Sum 1 output bit per clock cycle, using Dadda/Booth-style adders | [/verigpu/generation/mul_pipeline_cycle.py](/verigpu/generation/mul_pipeline_cycle.py) | 58.2 | 863 | 32 |
| Sum 2 output bit per clock cycle, using Dadda/Booth-style adders | [/verigpu/generation/mul_pipeline_cycle.py](/verigpu/generation/mul_pipeline_cycle.py) | 63.2 | 1244 | 16 |
| Sum 4 output bit per clock cycle, using Dadda/Booth-style adders | [/verigpu/generation/mul_pipeline_cycle.py](/verigpu/generation/mul_pipeline_cycle.py) | 74.8 | 2018 | 8 |
| Sum 8 output bit per clock cycle, using Dadda/Booth-style adders | [/verigpu/generation/mul_pipeline_cycle.py](/verigpu/generation/mul_pipeline_cycle.py) | 93.2 | 3429 | 4|

Overall, maybe "Sum 2 output bit per clock cycle, using Dadda/Booth-style adders" gives good propagation delay, and low area?
