#!/bin/bash

set -ex
set -o pipefail

bash test/behav/verilator_compile_proc.sh
# bash test/behav/verilator_compile_comp.sh
bash test/behav/float/verilator/float_mul_pipeline_test_vtor.sh
bash test/behav/float/verilator/float_add_pipeline_test_vtor.sh
