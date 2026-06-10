#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from lilypond2wav.parser import find_tempo, parse
from lilypond2wav.synth import SineSynthesizer
from lilypond2wav.wav_io import SAMPLE_RATE, write_wav

_DEFAULT_BPM = 120


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
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'error: file not found: {input_path}', file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text()

    if args.bpm is not None:
        bpm = args.bpm
    else:
        bpm = find_tempo(text) or _DEFAULT_BPM

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.wav')

    try:
        events = parse(text, bpm)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)

    if not events:
        print('warning: no notes found, writing silent WAV', file=sys.stderr)

    synth = SineSynthesizer()
    samples = synth.synthesize(events, SAMPLE_RATE)
    try:
        write_wav(samples, str(output_path))
    except OSError as e:
        print(f'error: cannot write "{output_path}": {e.strerror}',
              file=sys.stderr)
        sys.exit(1)

    print(f'written {output_path}  ({bpm} BPM, {len(events)} events, '
          f'{samples.shape[0] / SAMPLE_RATE:.2f}s)')


if __name__ == '__main__':
    main()
