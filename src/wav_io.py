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

import wave

import numpy as np

SAMPLE_RATE = 44100


def write_wav(samples: np.ndarray, path: str,
              sample_rate: int = SAMPLE_RATE) -> None:
    """
    Write float32 samples as 16-bit mono WAV to the specified 'path'.
    Normalizes to the full int16 range before writing.
    """
    peak = np.max(np.abs(samples))
    if peak > 0:
        normalized = samples / peak
    else:
        normalized = samples
    int_samples = (normalized * 32767).astype(np.int16)

    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
