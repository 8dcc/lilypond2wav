#
# Copyright 2026 8dcc. All Rights Reserved.
#
# This file is part of lilypond2wav.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys

# Debug output is enabled when LILYPOND2WAV_DEBUG is set to a non-empty value.
_DEBUG_ENABLED = bool(os.environ.get('LILYPOND2WAV_DEBUG'))


def dbg(msg: str) -> None:
    """
    Print a debug message to stderr, but only if the
    LILYPOND2WAV_DEBUG environment variable is set.
    """
    if _DEBUG_ENABLED:
        print(f'debug: {msg}', file=sys.stderr)


def log(msg: str) -> None:
    """
    Print an informational message to stdout.
    """
    print(msg)


def wrn(msg: str) -> None:
    """
    Print a warning message to stderr.
    """
    print(f'warning: {msg}', file=sys.stderr)


def err(msg: str) -> None:
    """
    Print an error message to stderr.
    """
    print(f'error: {msg}', file=sys.stderr)
