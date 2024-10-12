#!/bin/bash

check_for_failure() {
    "$@" || failure=1
}

location="$(cd "$(dirname "${0}")";pwd -P)"
root="$(cd "$(dirname "${location}/../..")";pwd -P)"

echo "Running pre-commit checks..."
check_for_failure pre-commit run --all-files
echo "-------------------------------------------------------------------------"

if ((failure)); then
    echo "Automated tests failed" >&2
    exit 1
else
    echo "Automated tests succeeded" >&2
    exit 0
fi
