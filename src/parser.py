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

import re
from dataclasses import dataclass
from fractions import Fraction

import ly.lex
import ly.lex.lilypond as lyl
import ly.pitch

from notes import pitch_to_freq
from utils import dbg, wrn


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
            wrn('double-dotted note not supported, using single dot')
            i += 1

    return current_length, dotted, i


def _read_pitch(tokens: list, i: int, note_token,
                pitch_reader) -> tuple[tuple | None, int]:
    """
    Read the pitch of 'note_token', consuming any following Octave
    tokens. Returns ((note_idx, alter, octave), new_i), or (None, i)
    if the note name is not recognised.
    """
    result = pitch_reader(str(note_token))
    if result is None:
        return None, i

    note_idx, alter = result
    octave = 0
    while i < len(tokens) and isinstance(tokens[i], lyl.Octave):
        s = str(tokens[i])
        octave += s.count("'") - s.count(",")
        i += 1

    return (note_idx, alter, octave), i


def _read_chord_pitches(tokens: list, i: int,
                        pitch_reader) -> tuple[list, int]:
    """
    Consume tokens from ChordStart up to and including ChordEnd.
    Returns (pitches, new_i); each pitch is (note_idx, alter, octave).
    """
    pitches = []
    while i < len(tokens) and not isinstance(tokens[i], lyl.ChordEnd):
        tok = tokens[i]
        i += 1
        if isinstance(tok, lyl.Note):
            pitch, i = _read_pitch(tokens, i, tok, pitch_reader)
            if pitch is None:
                wrn(f'unrecognised note name "{tok}"')
            else:
                pitches.append(pitch)
        elif not isinstance(tok, _IGNORABLE):
            wrn(f'skipping unsupported token '
                f'{type(tok).__name__} "{tok}" in chord')

    if i < len(tokens):
        i += 1  # consume ChordEnd

    return pitches, i


def _collect_pitches(tokens: list, i: int, t,
                     pitch_reader) -> tuple[list, int]:
    """
    Return (pitches, new_i) for a Note or ChordStart token 't'.
    Each pitch is (note_idx, alter, octave).
    """
    if isinstance(t, lyl.ChordStart):
        return _read_chord_pitches(tokens, i, pitch_reader)
    pitch, i = _read_pitch(tokens, i, t, pitch_reader)
    return ([pitch] if pitch is not None else []), i


def _resolve_relative(pitches: list, anchor: ly.pitch.Pitch
                      ) -> tuple[list, ly.pitch.Pitch]:
    """
    Resolve relative-mode 'pitches' against 'anchor'. The new anchor
    is the first resolved pitch, matching LilyPond semantics for both
    single notes and chords.
    """
    resolved = []
    reference = anchor
    for note_idx, alter, octave in pitches:
        p = ly.pitch.Pitch(note_idx, alter, octave)
        p.makeAbsolute(reference)
        resolved.append((p.note, p.alter, p.octave))
        reference = p

    new_anchor = anchor
    if resolved:
        note_idx, alter, octave = resolved[0]
        new_anchor = ly.pitch.Pitch(note_idx, alter, octave)

    return resolved, new_anchor


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
    dbg(f'parse: {bpm} BPM, {spb:.4f}s per beat')

    pitch_reader = ly.pitch.pitchReader('nederlands')
    notes: list[NoteEvent] = []

    in_relative = False
    relative_anchor: ly.pitch.Pitch | None = None
    current_length = 4
    dotted = False
    tie_pending = False
    current_time_s = 0.0

    tokens = _strip(list(ly.lex.state('lilypond').tokens(text)))
    dbg(f'parse: {len(tokens)} tokens after strip')
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
                dbg(f'language set to "{lang}"')

        elif isinstance(t, lyl.PitchCommand) and str(t) == r'\relative':
            relative_anchor, i = _read_anchor(tokens, i, pitch_reader)
            in_relative = True
            dbg(f'relative mode: anchor note={relative_anchor.note} '
                f'alter={relative_anchor.alter} '
                f'octave={relative_anchor.octave}')

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

        elif isinstance(t, (lyl.Note, lyl.ChordStart)):
            pitches, i = _collect_pitches(tokens, i, t, pitch_reader)
            if not pitches:
                if isinstance(t, lyl.Note):
                    wrn(f'unrecognised note name "{t}"')
                else:
                    wrn('empty chord, skipping')
                continue

            current_length, dotted, i = _read_length_dot(
                tokens, i, current_length, dotted)

            if in_relative and relative_anchor is not None:
                pitches, relative_anchor = _resolve_relative(
                    pitches, relative_anchor)

            freqs = [pitch_to_freq(n, a, o) for n, a, o in pitches]
            dur_s = (4.0 / current_length) * (1.5 if dotted else 1.0) * spb

            dbg(f'note @ {current_time_s:.3f}s: '
                f'{[round(f, 2) for f in freqs]} Hz, '
                f'len={current_length} dotted={dotted} dur={dur_s:.3f}s')

            if tie_pending:
                tie_pending = False
                unextended = [f for f in freqs
                              if not _try_extend(notes, f,
                                                 current_time_s, dur_s)]
                dbg(f'tie: extended {len(freqs) - len(unextended)} of '
                    f'{len(freqs)} pitch(es), {len(unextended)} new')
                if unextended:
                    wrn('tie between different pitches '
                        '(slur?), treating as separate notes')
                    notes.extend(NoteEvent(f, current_time_s, dur_s)
                                 for f in unextended)
            else:
                notes.extend(NoteEvent(f, current_time_s, dur_s)
                             for f in freqs)

            current_time_s += dur_s

        elif isinstance(t, lyl.Rest):
            current_length, dotted, i = _read_length_dot(
                tokens, i, current_length, dotted)
            dur_s = (4.0 / current_length) * (1.5 if dotted else 1.0) * spb
            dbg(f'rest @ {current_time_s:.3f}s: '
                f'len={current_length} dotted={dotted} dur={dur_s:.3f}s')
            current_time_s += dur_s

        elif isinstance(t, lyl.Tie):
            tie_pending = True

        elif not isinstance(t, _IGNORABLE):
            wrn(f'skipping unsupported token {type(t).__name__} "{t}"')

    dbg(f'parse: produced {len(notes)} note event(s)')
    return notes


def find_tempo(text: str) -> int | None:
    """
    Return BPM from the first \\tempo 4 = N directive, or None.
    """
    m = re.search(r'\\tempo\s+\d+\s*=\s*(\d+)', text)
    return int(m.group(1)) if m else None
