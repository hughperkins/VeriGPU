#!/bin/bash

set -ex
set -o pipefail

# first build the runtime and gpu...
prot/verilator/prot_single_source/build-cmake.sh

examples/cpp_single_source/data_transfer/run.sh
examples/cpp_single_source/sum_ints/run.sh
