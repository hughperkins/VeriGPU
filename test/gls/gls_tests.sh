#!/bin/bash

set -ex
set -o pipefail

bash test/gls/int_div_regfile_test_gls.sh
bash test/gls/mem_delayed_test_gls.sh
bash test/gls/int_div_regfile_comp_test_gls.sh
bash test/gls/proc_gls.sh
