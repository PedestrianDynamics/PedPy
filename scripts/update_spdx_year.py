# SPDX-FileCopyrightText: 2022-2026 Forschungszentrum Jülich GmbH, IAS-7
# SPDX-FileCopyrightText: 2024-2026 Tobias Schrödter
# SPDX-License-Identifier: MIT

"""Update the end year in SPDX-FileCopyrightText headers of modified files."""

import datetime
import pathlib
import re
import subprocess
import sys

_YEAR_RANGE_RE = re.compile(r"(SPDX-FileCopyrightText:\s*)(\d{4})(?:-(\d{4}))?(\s+)")


def _update_file_year(path: pathlib.Path, current_year: int) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    def _repl(match: re.Match[str]) -> str:
        prefix, start_str, end_str, suffix = match.groups()
        start_year = int(start_str)
        if start_year >= current_year:
            return match.group(0)
        if end_str is not None and int(end_str) >= current_year:
            return match.group(0)
        return f"{prefix}{start_year}-{current_year}{suffix}"

    updated = _YEAR_RANGE_RE.sub(_repl, content)
    if updated != content:
        path.write_text(updated, encoding="utf-8", newline="\n")
        return True
    return False


def main(argv: list[str]) -> int:
    """Update SPDX copyright end-years on passed paths or git-modified files."""
    current_year = datetime.date.today().year
    if argv:
        targets = [pathlib.Path(arg) for arg in argv]
    else:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "--cached"],
            text=True,
        ).splitlines()
        targets = [pathlib.Path(rel) for rel in changed]

    for target in targets:
        _update_file_year(target, current_year)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
