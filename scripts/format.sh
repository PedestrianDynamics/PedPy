#! /bin/bash

set -e

pre-commit run ruff-include-sorting --all-files
pre-commit run ruff-format --all-files
