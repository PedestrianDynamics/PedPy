#! /bin/bash

set -e

origin=$(dirname "$(readlink -f "$0")")
root=$(readlink -f "${origin}"/..)
isort --jobs "$(nproc)" "${root}"
black "${root}"