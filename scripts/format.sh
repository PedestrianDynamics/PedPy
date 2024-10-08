#! /bin/bash

set -e

location="$(cd "$(dirname "${0}")";pwd -P)"
root="$(cd "$(dirname "${location}/../..")";pwd -P)"

python -m ruff check --select I --fix "${root}"
python -m ruff format "${root}"