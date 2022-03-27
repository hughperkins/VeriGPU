# Coding guidelines

These coding guidelines aim principally to ensure that things that work in simulation also will work at tape-out.

A lot of errors are caught duing GLS ("gate-level simulation"), however these errors are often tricky and tedious to track down. Therefore a secondary concern of these guidelines is to enable rapid GLS test success.

There might also be some coding style guidelines included, but the initial goal is successful tape-out, and easy GLS.

Let's first go over some of the things that can go wrong; and then we will look at what guidelines we can use to avoid these issues.

## Things that can go wrong at GLS

### Firstly, a word on GLS

GLS testing is very 'black-boxy'. When something goes wrong, it's very hard to track it down. All our behavioral code is now in gates and cells, and it's very hard to figure out how or why the tests are failing.

### `if` and `case` statements behave inconsistently to `x` values

During behavioral testing, `if` statements which evaluate a condition that results in an `x` value choose the `else` path. This converts an `x` into a clean `0` or `1` effectively. For example, the following code converts an `x` value of `in` into a `0` value of `out`:

```
if(in) begin
    out = 1;
end else begin
    out = 0;
end
```

However, once the behavioral `if` has been converted into a combinatorial equivalent, any `x` that arrives at the combinatorial equivalent of the `if` propagates as an `x`. This means for example that a simple statement such as:

```
if(req) begin
    state = STATE2;
end
```

... in behavioral will simply not switch states if `req` is `x`. However, at GLS, this will assign `x` to `state`.


Similarly, for `case` statements, an `x` condition results in the `default` branch being selected, at behavioral, whilst leads to `x` propagation at GLS.

### Timing

In a unit-test test bench, for some sub-module, we might have something like:

```
#5 clk = 0;
#5 clk = 1;
req = 1;
```

In behavioral code, this will essentially simulate that some flip-flop output changed the value of `req` to `1`. And this works ok at behavioral. In GLS, this test will still pass. However, it no longer simulates the GLS version of a flip-flop. The GLS version of a flip-flop has a very slight propagation delay.

When we use this sub-module inside a larger module, and run GLS, the input to this sub-module no longer comes from the workbench changing exactly on the clock-tick, but comes from a GLS flip-flop, with a very slighlty delayed change. This can result in the test workbench on the larger module failing, in ways that are very hard to track down.


the test-bench on the larger module might now fail, and the input comes from an actual GLS flip-flop, the behavior is different from our test-bench behavior, and the GLS test fails.

## Coding guidelines to avoid GLS issues

### Why do we want to avoid GLS issues?

Tracking down errors in GLS tests is very tricky and time-consuming. All our behavioral code has now become gates and flip-flops, and it's very hard to see what is going on, and track down any issues.

### Avoiding `x` issues with `if` and `case`

Before any `if` or `case` condition, we use the macro ``asssert_known(.)`, defined in [src/assert.sv](/src/assert.sv), for each of the `reg`s used in the condition, to ensure that the `reg` is not `x`. This will capture any `x`s during behavioral simulation. This will ensure that at GLS, no unexpected `x`s will arrive at the `if` or `case` statements.

During GLS and timing, we compile against [src/assert_ignore.sv](/src/assert_ignore.sv), which simply replaces the ``assert_known(.)` macro with a `NOP`.

### Avoiding timing issues in our workbench

To simulate more closely the behavior of flip-flops at GLs, in our workbenches we use `<=` instead of `=`. This simulates that the change to the value is slightly after the rising clock-edge, not exactly on the clock-edge. That is instead of writing e.g.:

```
#5 clk = 0;
#5 clk = 1;
req = 1;
```
... we write:
```
#5 clk = 0;
#5 clk = 1;
req <= 1;
```

Empirically, this catches errors in unit-testing test-benches for sub-modules, such that if the unit-testing test-bench for the sub-module works, then the sub-module will behave as expected when used as part of a large module.

## Style guidelines

### `clk`, in test benches

In the workbench, if we use an automatic `forever` clk, it can be hard to keep track of which clock edge we are in. Consider the following code:
```
#5;
#5;
#5;
...
#5;
// are we now in positive or negative clk?
```

But coding `clk` manually is tedious and hard to read:
```
#5 clk = 0;
#5 clk = 1;
#5 clk = 0;
#5 clk = 1;
...
```

Therefore we use functions `tick()`, `pos()` and `neg()`. `tick()` is a negative clock edge followed by a positive clock edge. These are easy to read, easy to write, and we know immediately whether we are in positive or zero `clk`. 

```
tick();
tick();
...
tick():
// we are clearly in positive clk now.
```

### Generators

We use generators for creating Dadda adders, and Dadda-style adders, e.g. [verigpu/generation/mul_pipeline_cycle.py](verigpu/generation/mul_pipeline_cycle.py)
- The generated code should be stored itself in git, in [/src/generated](/src/generated).
- It is stored in a special directory, to keep it separate from non-generated code
- The generated files should have a header at the top. The header should explain that the file is generated, and should not be edited by hand. The header should give the path of the code used to generate it.
