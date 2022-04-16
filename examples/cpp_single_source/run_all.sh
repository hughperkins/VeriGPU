#!/bin/bash

set -ex
set -o pipefail

# first build the runtime and gpu...
bash prot/verilator/prot_single_source/build-cmake.sh

for ex in $(ls -b examples/cpp_single_source/*.cpp); do {
    bash examples/cpp_single_source/run.sh $(basename ${ex} .cpp)
} done
