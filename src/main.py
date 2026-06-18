#!/usr/bin/env python3
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

import argparse
import sys
from pathlib import Path

from parser import find_tempo, parse
from synth import HarmonicSynthesizer, SineSynthesizer
from utils import dbg, err, log, wrn
from wav_io import SAMPLE_RATE, write_wav

DEFAULT_BPM = 120

SYNTHESIZERS = {
    'harmonic': HarmonicSynthesizer,
    'sine': SineSynthesizer,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert a LilyPond file to WAV audio.')
    parser.add_argument('input', metavar='INPUT.ly',
                        help='LilyPond source file')
    parser.add_argument('-b', '--bpm', type=int, default=None,
                        help='Tempo in BPM (overrides \\tempo in file)')
    parser.add_argument('-o', '--output', metavar='OUTPUT.wav',
                        default=None,
                        help='Output WAV path (default: input stem + .wav)')
    parser.add_argument('-g', '--gate', type=float, default=1.0,
                        help='Note gate: fraction of each note that sounds '
                             '(0.0-1.0, default: 1.0)')
    parser.add_argument('-s', '--synth', choices=sorted(SYNTHESIZERS),
                        default='harmonic',
                        help='Synthesizer to use (default: harmonic)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.gate <= 0.0 or args.gate > 1.0:
        err('gate must be in the range (0.0, 1.0]')
        sys.exit(1)

    if not input_path.exists():
        err(f'file not found: {input_path}')
        sys.exit(1)

    text = input_path.read_text()

    if args.bpm is not None:
        bpm = args.bpm
        bpm_source = '--bpm'
    else:
        found = find_tempo(text)
        bpm = found or DEFAULT_BPM
        bpm_source = r'\tempo' if found else 'default'
    dbg(f'tempo: {bpm} BPM (from {bpm_source})')

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.wav')
    dbg(f'input: {input_path}, output: {output_path}')

    try:
        notes = parse(text, bpm)
    except ValueError as e:
        err(str(e))
        sys.exit(1)

    if not notes:
        wrn('no notes found, writing silent WAV')

    dbg(f'synthesizing with {args.synth} (gate={args.gate})')
    synth = (SYNTHESIZERS[args.synth])()
    samples = synth.synthesize(notes, SAMPLE_RATE, gate=args.gate)
    try:
        write_wav(samples, str(output_path))
    except OSError as e:
        err(f'cannot write "{output_path}": {e.strerror}')
        sys.exit(1)

    log(f'written {output_path}  ({bpm} BPM, {len(notes)} notes, '
        f'{samples.shape[0] / SAMPLE_RATE:.2f}s)')


if __name__ == '__main__':
    main()
