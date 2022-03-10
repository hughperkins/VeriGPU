#!/bin/bash

(cd docker
docker build -t chip_design .
docker push hughperkins/chip_design:latest
)
