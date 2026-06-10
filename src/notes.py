from fractions import Fraction

# Semitone offsets for C D E F G A B
_SEMITONES = [0, 2, 4, 5, 7, 9, 11]


def pitch_to_freq(note: int, alter: Fraction, octave: int) -> float:
    """
    Return frequency in Hz for the given 'note' (diatonic index 0-6,
    C=0 through B=6), 'alter' (0=natural, 1/2=sharp, -1/2=flat), and
    'octave' in the ly.pitch convention (c'=1 is middle C, MIDI 60).
    """
    midi = 48 + _SEMITONES[note] + round(alter * 2) + 12 * octave
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)
