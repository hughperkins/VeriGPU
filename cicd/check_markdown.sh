#!/bin/bash

# pre-requisites:
# npm i --global -D markdown-link-check
# page: https://github.com/tcort/markdown-link-check

find . -name \*.md -print0 | xargs -0 -n1 markdown-link-check -q -c cicd/markdown-link-check-config.json 
