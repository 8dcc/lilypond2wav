import re
import sys
from dataclasses import dataclass
from fractions import Fraction

import ly.lex
import ly.lex.lilypond as lyl
import ly.pitch

from notes import pitch_to_freq


@dataclass
class NoteEvent:
    frequency: float    # Hz
    start_s: float      # absolute start time in seconds
    duration_s: float   # full duration in seconds (before gate)


# Map language names not known to ly.pitch to their canonical equivalent.
# ly.pitch supports all LilyPond languages except accented variants.
_LANGUAGE_ALIASES: dict[str, str] = {
    'català': 'catalan',
    'español': 'espanol',
    'português': 'portugues',
}


# Tokens that carry no musical meaning; skip silently.
#
# NOTE: \key c \major produces PitchCommand + Note + KeySignatureMode;
# the \key handler consumes the Note so it never reaches the Note
# branch. All other structural tokens are listed here.
_IGNORABLE = (
    ly.lex.Space,
    ly.lex.Comment,
    lyl.SequentialStart,
    lyl.SequentialEnd,
    lyl.PipeSymbol,
    lyl.EqualSign,
    lyl.IntegerValue,
    lyl.StringQuotedStart,
    lyl.StringQuotedEnd,
    lyl.String,
    lyl.Keyword,           # \version, \language (handled above), etc.
    lyl.Command,           # \time, \bar, \mark, etc.
    lyl.Fraction,          # 4/4 after \time
    lyl.Header,            # \header
    lyl.Score,             # \score
    lyl.New,               # \new
    lyl.ContextName,       # Staff, Voice, etc.
    lyl.Clef,              # \clef
    lyl.ClefSpecifier,     # treble, bass, etc.
    lyl.OpenBracket,       # { after \header / \score
    lyl.CloseBracket,      # } after \header / \score
    lyl.HeaderVariable,    # tagline, title, etc.
    lyl.KeySignatureMode,  # \major, \minor
    lyl.Beam,              # [ and ] (manual beams, engraving only)
)


def _strip(tokens: list) -> list:
    """
    Remove whitespace and comment tokens.
    """
    skip = (ly.lex.Space, ly.lex.Comment)
    return [t for t in tokens if not isinstance(t, skip)]


def _read_anchor(tokens: list, i: int,
                 pitch_reader) -> tuple[ly.pitch.Pitch, int]:
    """
    Read optional anchor note + octave marks after \\relative.
    Returns (anchor_pitch, new_i). Stops before SequentialStart.
    """
    anchor = ly.pitch.Pitch(0, Fraction(0), 1)  # default: c' = C4

    if i < len(tokens) and isinstance(tokens[i], lyl.Note):
        result = pitch_reader(str(tokens[i]))
        if result:
            note_idx, alter = result
            i += 1
            octave = 0
            while i < len(tokens):
                tok = tokens[i]
                if isinstance(tok, lyl.SequentialStart):
                    break
                s = str(tok)
                if set(s) <= {"'", ","}:
                    octave += s.count("'") - s.count(",")
                    i += 1
                else:
                    break
            anchor = ly.pitch.Pitch(note_idx, alter, octave)

    return anchor, i


def _read_length_dot(tokens: list, i: int, current_length: int,
                     dotted: bool) -> tuple[int, bool, int]:
    """
    Consume optional Length and Dot tokens starting at i.
    Returns (new_current_length, new_dotted, new_i).
    """
    if i < len(tokens) and isinstance(tokens[i], lyl.Length):
        current_length = int(str(tokens[i]))
        i += 1

    dotted = False
    if i < len(tokens) and isinstance(tokens[i], lyl.Dot):
        dotted = True
        i += 1
        if i < len(tokens) and isinstance(tokens[i], lyl.Dot):
            print('warning: double-dotted note not supported, '
                  'using single dot', file=sys.stderr)
            i += 1

    return current_length, dotted, i


def _try_extend(notes: list, freq: float,
                at_s: float, by_s: float) -> bool:
    """
    Search 'notes' in reverse for one ending at 'at_s' with a
    matching 'freq', extending its duration by 'by_s' if found.
    """
    for note in reversed(notes):
        if note.start_s + note.duration_s < at_s - 1e-9:
            break
        if abs(note.frequency - freq) < 0.01:
            note.duration_s += by_s
            return True
    return False


def parse(text: str, bpm: int) -> list[NoteEvent]:
    """
    Parse the specified LilyPond 'text' and return a flat list of
    notes with absolute start times. 'bpm' is beats per minute
    (quarter note = 1 beat). Silence is implicit (absence of notes).
    """
    spb = 60.0 / bpm  # seconds per beat

    pitch_reader = ly.pitch.pitchReader('nederlands')
    notes: list[NoteEvent] = []

    in_relative = False
    relative_anchor: ly.pitch.Pitch | None = None
    current_length = 4
    dotted = False
    tie_pending = False
    current_time_s = 0.0

    tokens = _strip(list(ly.lex.state('lilypond').tokens(text)))
    i = 0

    while i < len(tokens):
        t = tokens[i]
        i += 1

        if isinstance(t, lyl.Keyword) and str(t) == r'\language':
            # StringQuotedStart/End are subclasses of lyl.String, so
            # use exact type match to find the content token only.
            while i < len(tokens) and type(tokens[i]) is not lyl.String:
                i += 1

            if i < len(tokens):
                lang = str(tokens[i])
                i += 1
                lang = _LANGUAGE_ALIASES.get(lang, lang)
                try:
                    pitch_reader = ly.pitch.pitchReader(lang)
                except Exception:
                    raise ValueError(f'unknown language "{lang}"')

        elif isinstance(t, lyl.PitchCommand) and str(t) == r'\relative':
            relative_anchor, i = _read_anchor(tokens, i, pitch_reader)
            in_relative = True

        elif isinstance(t, lyl.PitchCommand) and str(t) == r'\key':
            # skip: Note KeySignatureMode
            while i < len(tokens) and isinstance(
                    tokens[i], (lyl.Note, lyl.KeySignatureMode)):
                i += 1

        elif isinstance(t, lyl.PitchCommand):
            pass  # other PitchCommand (e.g. \trill) -- silently skip

        elif isinstance(t, lyl.Tempo):
            # skip: Length EqualSign IntegerValue
            while i < len(tokens) and isinstance(tokens[i], (
                    lyl.Length, lyl.EqualSign, lyl.IntegerValue)):
                i += 1

        elif isinstance(t, lyl.SequentialEnd) and in_relative:
            in_relative = False
            relative_anchor = None

        elif isinstance(t, lyl.Note):
            result = pitch_reader(str(t))
            if result is None:
                print(f'warning: unrecognised note name "{t}"', file=sys.stderr)
                continue

            note_idx, alter = result
            octave = 0
            while i < len(tokens) and isinstance(tokens[i], lyl.Octave):
                s = str(tokens[i])
                octave += s.count("'") - s.count(",")
                i += 1

            current_length, dotted, i = _read_length_dot(
                tokens, i, current_length, dotted)

            if in_relative and relative_anchor is not None:
                p = ly.pitch.Pitch(note_idx, alter, octave)
                p.makeAbsolute(relative_anchor)
                relative_anchor = p.copy()
                octave = p.octave
                alter = p.alter

            freq = pitch_to_freq(note_idx, alter, octave)
            dur_s = (4.0 / current_length) * (1.5 if dotted else 1.0) * spb

            if tie_pending:
                tie_pending = False
                if not _try_extend(notes, freq, current_time_s, dur_s):
                    print('warning: tie between different pitches '
                          '(slur?), treating as separate notes',
                          file=sys.stderr)
                    notes.append(NoteEvent(freq, current_time_s, dur_s))
            else:
                notes.append(NoteEvent(freq, current_time_s, dur_s))

            current_time_s += dur_s

        elif isinstance(t, lyl.Rest):
            current_length, dotted, i = _read_length_dot(
                tokens, i, current_length, dotted)
            dur_s = (4.0 / current_length) * (1.5 if dotted else 1.0) * spb
            current_time_s += dur_s

        elif isinstance(t, lyl.Tie):
            tie_pending = True

        elif not isinstance(t, _IGNORABLE):
            print(f'warning: skipping unsupported token '
                  f'{type(t).__name__} "{t}"', file=sys.stderr)

    return notes


def find_tempo(text: str) -> int | None:
    """
    Return BPM from the first \\tempo 4 = N directive, or None.
    """
    m = re.search(r'\\tempo\s+\d+\s*=\s*(\d+)', text)
    return int(m.group(1)) if m else None
