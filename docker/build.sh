#!/bin/bash

set -x
set -e

(cd docker
sudo docker build -t hughperkins/chip_design:latest .
sudo docker push hughperkins/chip_design:latest
)
