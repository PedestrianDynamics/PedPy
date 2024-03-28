#! /bin/bash

check_for_failure() {
    "$@" || failure=1
}

location="$(cd "$(dirname "${0}")";pwd -P)"
root="$(cd "$(dirname "${location}/../..")";pwd -P)"

echo "Check format"
check_for_failure "${root}"/scripts/check-format.sh
echo "-------------------------------------------------------------------------"

echo "Check docstring style"
check_for_failure pydocstyle
echo "-------------------------------------------------------------------------"

echo "Check typing with mypy"
check_for_failure python3 -m mypy --config-file mypy.ini pedpy/
echo "-------------------------------------------------------------------------"

echo "Linting with pylint"
check_for_failure python3 -m pylint --recursive=y --extension-pkg-whitelist=scipy pedpy pedpy/data pedpy/io pedpy/methods pedpy/plotting pedpy/internal
echo "-------------------------------------------------------------------------"

if ((failure)); then
    echo "Automated tests failed" >&2
    exit 1
else
    echo "Automated tests succeeded" >&2
    exit 0
fi
