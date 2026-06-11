#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from parser import find_tempo, parse
from synth import HarmonicSynthesizer, SineSynthesizer
from wav_io import SAMPLE_RATE, write_wav

_DEFAULT_BPM = 120

_SYNTHESIZERS = {
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
    parser.add_argument('-s', '--synth', choices=sorted(_SYNTHESIZERS),
                        default='harmonic',
                        help='Synthesizer to use (default: harmonic)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.gate <= 0.0 or args.gate > 1.0:
        print('error: gate must be in the range (0.0, 1.0]', file=sys.stderr)
        sys.exit(1)

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

    synth = _SYNTHESIZERS[args.synth]()
    samples = synth.synthesize(events, SAMPLE_RATE, gate=args.gate)
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
