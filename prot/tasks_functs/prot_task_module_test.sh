#!/bin/bash

BASE=prot/tasks_functs

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv ${BASE}/prot_task.sv ${BASE}/prot_task_module.sv ${BASE}/prot_task_module_test.sv
./a.out
