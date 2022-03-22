# Verification guidelines

## Unit tests

Each module should have a unit test for it. Currently these tests are written using verilog test-bench scripts, though this is not necessarily set in stone.

## Propagation delay testing

Each module should be tested for propagation delay, using [toy_proc/timing.py](/toy_proc/timing.py). The result should be added to a comment at the top of the document. If the module is modified, the timing should be re-run.

If the timing script fails to run, this is likely because of unitialized `reg`s that are being used in the combinatorial `always` block. If you open up `build/netlist/6.v`, and search for `always`, you can usually find hints as to which `reg`s were not initialized for all execution paths.

## Gate-level simulation (GLS)

Each module should be tested using GLS, as well as using the behavioral tests. GLS catches issues with timing, and with conditions in `if` and `case`, which might not be caught in the behavioral tests.
