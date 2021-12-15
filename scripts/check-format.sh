#! /bin/bash

origin=$(dirname "$(readlink -f "$0")")
root=$(readlink -f "${origin}/..")

return_code=0

isort --check --jobs "$(nproc)" "${root}" || return_code=1
black --check "${root}" || return_code=1

exit ${return_code}