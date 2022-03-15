#!/bin/bash

set -e
set -x
set -o pipefail

for prog in prog2 prog3 prog4 prog5 prog6 prog7 prog8 prog9 prog10 prog11 prog12 \
        prog13 prog14 prog15 prog16 prog17 prog18 prog19 prog20 prog21 prog22; do {
    python run.py --name ${prog}
} done
