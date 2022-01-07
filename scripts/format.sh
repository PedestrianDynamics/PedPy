#! /bin/bash

set -e

location="$(cd "$(dirname "${0}")";pwd -P)"
root="$(cd "$(dirname "${location}/../..")";pwd -P)"

njobs=1
if [[ "$(uname)" == "Darwin" ]]; then
    njobs=$(sysctl -n hw.logicalcpu)
elif [[ "$(uname)" == "Darwin" ]]; then
    njobs=$(nproc)
fi

isort --jobs "${njobs}" "${root}"
black "${root}"