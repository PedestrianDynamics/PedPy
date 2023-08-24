#! /bin/bash

location="$(cd "$(dirname "${0}")";pwd -P)"
root="$(cd "$(dirname "${location}/../..")";pwd -P)"

echo "Check format"
"${root}"/scripts/check-format.sh
echo "-------------------------------------------------------------------------"

echo "Check docstring style"
pydocstyle
echo "-------------------------------------------------------------------------"

echo "Check typing with mypy"
python3 -m mypy --config-file mypy.ini pedpy/
echo "-------------------------------------------------------------------------"

echo "Linting with pylint"
python3 -m pylint --recursive=y --extension-pkg-whitelist=scipy pedpy pedpy/data pedpy/io pedpy/methods pedpy/plotting

