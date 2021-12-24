#! /bin/bash

set -e

location="$(cd "$(dirname "${0}")";pwd -P)"
root=$(readlink -f "${location}"/..)
isort --jobs "$(nproc)" "${root}"
black "${root}"