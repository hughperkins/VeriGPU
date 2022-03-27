#!/bin/bash

set -ex
set -o pipefail

python verigpu/generation/mul_pipeline_cycle.py --width 32 --bits-per-cycle 2 --out-dir src/generated
python verigpu/generation/mul_pipeline_cycle.py --width 24 --bits-per-cycle 2 --out-dir src/generated

python verigpu/generation/dadda.py --out-dir src/generated --width 24 --out-width 48
python verigpu/generation/dadda.py --out-dir src/generated --width 32 --out-width 32
