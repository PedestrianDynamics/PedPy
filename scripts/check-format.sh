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

python -m ruff check --select I "${root}" || return_code=1
ruff format --check "${root}" || return_code=1

exit ${return_code}
