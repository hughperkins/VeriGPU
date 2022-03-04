#!/bin/bash

set -e
set -x

for prog in prog2 prog3 prog4 prog5 prog6; do {
    python run.py --name ${prog}
} done
