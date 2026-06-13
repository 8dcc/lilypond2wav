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

from fractions import Fraction

# Semitone offsets for C D E F G A B
_SEMITONES = [0, 2, 4, 5, 7, 9, 11]


def pitch_to_freq(note: int, alter: Fraction, octave: int) -> float:
    """
    Return frequency in Hz for the given 'note' (diatonic index 0-6,
    C=0 through B=6), 'alter' (0=natural, 1/2=sharp, -1/2=flat), and
    'octave' in the ly.pitch convention (c'=1 is middle C, MIDI 60).
    """
    diatonic_semitones = _SEMITONES[note]
    accidental_semitones = round(alter * 2)
    midi = 48 + diatonic_semitones + accidental_semitones + 12 * octave
    semitones_from_a4 = midi - 69
    return 440.0 * 2.0 ** (semitones_from_a4 / 12.0)
