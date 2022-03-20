#!/bin/bash

# runs GLS on entire comp.sv, including memory etc
# (Slow :D so we split it out into separate CICD build)

# we need to somehow make this only run from time to time
# not sure how to do that. let's just not run it automatically for now?

bash test/gls/comp_gls.sh
