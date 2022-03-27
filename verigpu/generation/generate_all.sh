#!/bin/bash

set -ex
set -o pipefail

for bpc in 1 2 4 8; do {
    python verigpu/generation/mul_pipeline_cycle.py --width 32 --bits-per-cycle ${bpc} --out-dir src/generated
} done

python verigpu/generation/dadda.py --out-dir src/generated --width 24 --out-width 48
python verigpu/generation/dadda.py --out-dir src/generated --width 32 --out-width 32
