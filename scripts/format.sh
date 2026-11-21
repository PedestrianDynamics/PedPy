#! /bin/bash
# SPDX-FileCopyrightText: 2022-2026 Forschungszentrum Jülich GmbH, IAS-7
# SPDX-FileCopyrightText: 2024-2026 Tobias Schrödter
# SPDX-License-Identifier: MIT

set -e

location="$(cd "$(dirname "${0}")";pwd -P)"
root="$(cd "$(dirname "${location}/../..")";pwd -P)"

python -m ruff check --select I --fix "${root}"
python -m ruff format "${root}"
