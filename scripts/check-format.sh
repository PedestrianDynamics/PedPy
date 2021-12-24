#! /bin/bash

location="$(cd "$(dirname "${0}")";pwd -P)"
root=$(readlink -f "${location}/..")

return_code=0

isort --check --jobs "$(nproc)" "${root}" || return_code=1
black --check "${root}" || return_code=1

exit ${return_code}