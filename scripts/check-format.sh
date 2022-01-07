#! /bin/bash

location="$(cd "$(dirname "${0}")";pwd -P)"
root="$(cd "$(dirname "${location}/../..")";pwd -P)"

njobs=1
if [[ "$(uname)" == "Darwin" ]]; then
    njobs=$(sysctl -n hw.logicalcpu)
elif [[ "$(uname)" == "Darwin" ]]; then
    njobs=$(nproc)
fi


return_code=0

isort --check --jobs "${njobs}" "${root}" || return_code=1
black --check "${root}" || return_code=1

exit ${return_code}