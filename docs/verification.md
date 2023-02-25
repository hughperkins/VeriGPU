# Verification

We run the following verification:
- behavioral-level simulation and testing
- gate-level simulation and testing
- unit-tests for the assembler, and for the test runner scripts themselves

## Continuous Integration (CI)

[![CircleCI](https://circleci.com/gh/hughperkins/VeriGPU/tree/main.svg?style=svg)](https://circleci.com/gh/hughperkins/VeriGPU/tree/main)

The CI server runs the following verification scripts:
- python: [/cicd/run-py.sh](/cicd/run-py.sh).
- behavioral: [/cicd/run-behav.sh](/cicd/run-behav.sh).
- gate-level simulation: [/cicd/run-gls.sh](/cicd/run-gls.sh).

## Behavioral-level simulation and testing

### Concept

We use the `iverilog` simulator to run simulation of our design at the behavioral level. This is the first stage of verification, since these simulations run fastest, and are easiest to debug.

### Pre-requisites

- `python3`
- `iverilog`
- have cloned this repo, and be at the root of this repo

### Procedure

```
bash test/behav/run_sv_tests.sh
```

This will run various behavioral-level unit-tests at [/test/behav](/test/behav). The script is [/test/behav/run_sv_tests.sh](/test/behav/run_sv_unit_tests.sh).

```
bash test/behav/run_examples.sh
```

- under the hood, this will run many of the examples in [examples/direct](/examples/direct), and check the outputs against the expected outputs, which are in the paired files, with suffix `_expected.txt`, in the sub-folder [examples/direct/expected](/examples/direct/expected) folder.

## Gate-level simulation and testing

### Concept

Gate-level simulation runs on partially synthesized verilog, where the behavioral-level code, such as `always` blocks and `if` blocks has been converted into purely combinatorial logic and flip-flops, and then further converted into actual cells, using osu018 technology.

We use [yosys](https://yosyshq.net/yosys/) to simulate down to cell-level, using asu018 technology, then use [iverilog](http://iverilog.icarus.com/) to run simulation of the resulting netlist.

### Results

You can view execution of these tests in the CI server linked above.

### Pre-requisites

- `python3`
- [yosys](https://yosyshq.net/yosys/)
- [iverilog](http://iverilog.icarus.com/)
- have cloned this repo, and be in the root of this cloned repo

### Procedure

```
bash test/gls/gls_tests.sh
```

See [test/gls/gls_tests.sh](/test/gls/gls_tests.sh).


## Unit-tests for the assembler and some of the test scripts themselves

### Concept

The assembler is written in python, as are some of the test scripts, such as for timing. We use [pytest](https://docs.pytest.org/en/7.1.x/) to verify these scripts.

### Pre-requities

- have python3
- from this repo, have done:
    - `pip install -e .`
    - `pip install -r test/py/requirements.txt`

### Procedure

```
pytest -v
```

## Verification guidelines

See [docs/verification_guidelines.md](/docs/verification_guidelines.md).

## Coding guidelines

In order to ensure succesful gate-level simulation and tape-out, we have created coding guidelines at [docs/coding_guidelines.md](/docs/coding_guidelines.md)
