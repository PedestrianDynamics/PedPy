#! /bin/bash

return_code=0

pre-commit run --hook-stage manual ruff-include-sorting-dry -a || return_code=1
pre-commit run --hook-stage manual ruff-format-dry -a || return_code=1

exit ${return_code}
