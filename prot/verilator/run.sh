#!/bin/bash

set -ex
set -o pipefail

cd prot/verilator

if [[ ! -d build ]]; then {
    mkdir build
} fi

cd build

cmake ..
make -j $(nproc)
./verilator1

# BASE=../../../src
# for file in ${BASE}/comp.sv ${BASE}/op_const.sv ${BASE}/assert.sv ${BASE}/const.sv \
#     ${BASE}/float/float_params.sv ${BASE}/float/float_add_pipeline.sv \
#     ${BASE}/int/chunked_add_task.sv ${BASE}/int/chunked_sub_task.sv \
#     ${BASE}/generated/mul_pipeline_cycle_24bit_2bpc.sv ${BASE}/float/float_mul_pipeline.sv \
#     ${BASE}/int/mul_pipeline_32bit.sv ${BASE}/generated/mul_pipeline_cycle_32bit_2bpc.sv \
#     ${BASE}/int/int_div_regfile.sv \
#     ${BASE}/proc.sv \
#     ${BASE}/mem_delayed_small.sv ${BASE}/mem_delayed.sv; do {
#     verilator -sv -cc  ${file}
# } done
# verilator -sv -cc  ${BASE}/comp.sv ${BASE}/op_const.sv ${BASE}/assert.sv ${BASE}/const.sv \
#     ${BASE}/float/float_params.sv ${BASE}/float/float_add_pipeline.sv \
#     ${BASE}/int/chunked_add_task.sv ${BASE}/int/chunked_sub_task.sv \
#     ${BASE}/generated/mul_pipeline_cycle_24bit_2bpc.sv ${BASE}/float/float_mul_pipeline.sv \
#     ${BASE}/int/mul_pipeline_32bit.sv ${BASE}/generated/mul_pipeline_cycle_32bit_2bpc.sv \
#     ${BASE}/int/int_div_regfile.sv \
#     ${BASE}/proc.sv \
#     ${BASE}/mem_delayed_small.sv ${BASE}/mem_delayed.sv
