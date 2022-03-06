#!/bin/bash

set -e
set -x

for prog in prog2 prog3 prog4 prog5 prog6 prog7 prog8 prog9 prog10 prog11 prog12 prog13 prog14; do {
    python run.py --name ${prog}
} done
