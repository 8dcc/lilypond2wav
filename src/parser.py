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


@dataclass
class SequenceState:
    """
    Mutable state for a single music sequence. Tracks the active note
    length, dot modifier, relative-mode anchor, and pending tie.
    """
    current_length: int = 4
    dotted: bool = False
    in_relative: bool = False
    relative_anchor: ly.pitch.Pitch | None = None
    tie_pending: bool = False


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


class PitchReader:
    """
    Wraps ly.pitch note-name parsing and relative-pitch resolution.
    The active language can be changed at any time via set_language().
    """

    _ALIASES: dict[str, str] = {
        'català': 'catalan',
        'español': 'espanol',
        'português': 'portugues',
    }

    def __init__(self) -> None:
        self._reader = ly.pitch.pitchReader('nederlands')

    def set_language(self, lang: str) -> None:
        """
        Switch the active note-name language. Raises ValueError for
        unknown languages.
        """
        lang = self._ALIASES.get(lang, lang)
        try:
            self._reader = ly.pitch.pitchReader(lang)
        except Exception:
            raise ValueError(f'unknown language "{lang}"')
        dbg(f'language set to "{lang}"')

    def read(self, note_name: str) -> tuple | None:
        """
        Map 'note_name' to (note_idx, alter), or None if unrecognised.
        """
        result = self._reader(note_name)
        return result if result is not False else None

    def resolve_relative(self, pitches: list,
                         anchor: ly.pitch.Pitch
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


class TokenCursor:
    """
    Wraps a stripped token list and advances through it. Provides
    generic traversal primitives and domain-specific readers.
    """

    def __init__(self, tokens: list) -> None:
        self._tokens = tokens
        self._i = 0

    def at_end(self) -> bool:
        return self._i >= len(self._tokens)

    def peek(self) -> ly.lex.Token | None:
        """Return the current token without consuming it, or None."""
        if self.at_end():
            return None
        return self._tokens[self._i]

    def consume(self) -> ly.lex.Token:
        """Return the current token and advance."""
        t = self._tokens[self._i]
        self._i += 1
        return t

    def read_if(self, token_type) -> ly.lex.Token | None:
        """Consume and return if current token matches type, else None."""
        if not self.at_end() and isinstance(self._tokens[self._i], token_type):
            return self.consume()
        return None

    def skip_while(self, token_types) -> None:
        """Advance past any tokens matching the given types."""
        while not self.at_end() and isinstance(
                self._tokens[self._i], token_types):
            self._i += 1

    def read_length_dot(self) -> tuple[int | None, bool]:
        """
        Consume optional Length and Dot tokens.
        Returns (length, dotted); length is None if no Length token seen.
        """
        length = None
        tok = self.read_if(lyl.Length)
        if tok is not None:
            length = int(str(tok))

        dotted = False
        if self.read_if(lyl.Dot) is not None:
            dotted = True
            if self.read_if(lyl.Dot) is not None:
                wrn('double-dotted note not supported, using single dot')

        return length, dotted

    def read_pitch(self, pitch_reader: PitchReader,
                   note_name: str) -> tuple | None:
        """
        Read pitch for 'note_name', consuming any following Octave tokens.
        Returns (note_idx, alter, octave), or None if unrecognised.
        """
        result = pitch_reader.read(note_name)
        if result is None:
            return None

        note_idx, alter = result
        octave = 0
        while not self.at_end() and isinstance(
                self._tokens[self._i], lyl.Octave):
            s = str(self.consume())
            octave += s.count("'") - s.count(",")

        return (note_idx, alter, octave)

    def read_chord_pitches(self, pitch_reader: PitchReader) -> list:
        """
        Consume tokens up to and including ChordEnd.
        Returns a list of (note_idx, alter, octave) tuples.
        """
        pitches = []
        while not self.at_end() and not isinstance(
                self._tokens[self._i], lyl.ChordEnd):
            tok = self.consume()
            if isinstance(tok, lyl.Note):
                pitch = self.read_pitch(pitch_reader, str(tok))
                if pitch is None:
                    wrn(f'unrecognised note name "{tok}"')
                else:
                    pitches.append(pitch)
            elif not isinstance(tok, _IGNORABLE):
                wrn(f'skipping unsupported token '
                    f'{type(tok).__name__} "{tok}" in chord')

        self.read_if(lyl.ChordEnd)
        return pitches

    def read_anchor(self, pitch_reader: PitchReader) -> ly.pitch.Pitch:
        """
        Read optional anchor note + octave marks after \\relative.
        Returns a Pitch (defaults to c' = C4 if no anchor present).
        """
        anchor = ly.pitch.Pitch(0, Fraction(0), 1)  # default: c' = C4

        if not self.at_end() and isinstance(self._tokens[self._i], lyl.Note):
            note_tok = self.consume()
            result = pitch_reader.read(str(note_tok))
            if result:
                note_idx, alter = result
                octave = 0
                while not self.at_end():
                    tok = self._tokens[self._i]
                    if isinstance(tok, lyl.SequentialStart):
                        break
                    s = str(tok)
                    if set(s) <= {"'", ","}:
                        octave += s.count("'") - s.count(",")
                        self._i += 1
                    else:
                        break
                anchor = ly.pitch.Pitch(note_idx, alter, octave)

        return anchor


class Parser:
    """
    Stateful LilyPond parser. Owns a PitchReader and the accumulated
    output note list for a single parse call.
    """

    def __init__(self, bpm: int) -> None:
        self._bpm = bpm
        self.pitch_reader = PitchReader()
        self.spb = 60.0 / bpm
        self.notes: list[NoteEvent] = []

    def _duration_s(self, state: SequenceState) -> float:
        factor = 1.5 if state.dotted else 1.0
        return (4.0 / state.current_length) * factor * self.spb

    def _try_extend(self, freq: float, at_s: float, by_s: float) -> bool:
        """
        Search 'self.notes' in reverse for one ending at 'at_s' with a
        matching 'freq', extending its duration by 'by_s' if found.
        """
        for note in reversed(self.notes):
            if note.start_s + note.duration_s < at_s - 1e-9:
                break
            if abs(note.frequency - freq) < 0.01:
                note.duration_s += by_s
                return True
        return False

    def _set_language(self, cursor: TokenCursor) -> None:
        """
        Handle \\language: read the quoted language name from the cursor
        and update 'self.pitch_reader'.
        """
        # StringQuotedStart/End are subclasses of lyl.String, so use exact
        # type match to find the content token only.
        while not cursor.at_end() and type(cursor.peek()) is not lyl.String:
            cursor.consume()
        if not cursor.at_end():
            self.pitch_reader.set_language(str(cursor.consume()))

    def _append_notes(self, freqs: list, cursor_s: float,
                      dur_s: float, state: SequenceState) -> None:
        """
        Append NoteEvents for 'freqs' at 'cursor_s', merging a pending
        tie if one is set on 'state'.
        """
        if state.tie_pending:
            state.tie_pending = False
            unextended = [f for f in freqs
                          if not self._try_extend(f, cursor_s, dur_s)]
            dbg(f'tie: extended {len(freqs) - len(unextended)} of '
                f'{len(freqs)} pitch(es), {len(unextended)} new')
            if unextended:
                wrn('tie between different pitches '
                    '(slur?), treating as separate notes')
                self.notes.extend(NoteEvent(f, cursor_s, dur_s)
                                  for f in unextended)
        else:
            self.notes.extend(NoteEvent(f, cursor_s, dur_s) for f in freqs)

    def _emit_note(self, cursor: TokenCursor, t,
                   state: SequenceState, cursor_s: float) -> float:
        """
        Collect, resolve, and emit NoteEvent(s) for Note or ChordStart
        token 't'. Returns the new cursor time.
        """
        if isinstance(t, lyl.ChordStart):
            pitches = cursor.read_chord_pitches(self.pitch_reader)
        else:
            pitch = cursor.read_pitch(self.pitch_reader, str(t))
            pitches = [pitch] if pitch is not None else []

        if not pitches:
            if isinstance(t, lyl.Note):
                wrn(f'unrecognised note name "{t}"')
            else:
                wrn('empty chord, skipping')
            return cursor_s

        length, dotted = cursor.read_length_dot()
        if length is not None:
            state.current_length = length
        state.dotted = dotted

        if state.in_relative and state.relative_anchor is not None:
            pitches, state.relative_anchor = \
                self.pitch_reader.resolve_relative(
                    pitches, state.relative_anchor)

        freqs = [pitch_to_freq(n, a, o) for n, a, o in pitches]
        dur_s = self._duration_s(state)

        dbg(f'note @ {cursor_s:.3f}s: '
            f'{[round(f, 2) for f in freqs]} Hz, '
            f'len={state.current_length} dotted={state.dotted} '
            f'dur={dur_s:.3f}s')

        self._append_notes(freqs, cursor_s, dur_s, state)
        return cursor_s + dur_s

    def _parse_tokens(self, cursor: TokenCursor,
                      state: SequenceState, start_s: float = 0.0) -> float:
        """
        Parse tokens from 'cursor' starting at 'start_s'. Returns end
        time in seconds.
        """
        cursor_s = start_s

        while not cursor.at_end():
            t = cursor.consume()

            if isinstance(t, lyl.Keyword) and str(t) == r'\language':
                self._set_language(cursor)

            elif isinstance(t, lyl.PitchCommand) and str(t) == r'\relative':
                state.relative_anchor = cursor.read_anchor(self.pitch_reader)
                state.in_relative = True
                dbg(f'relative mode: '
                    f'anchor note={state.relative_anchor.note} '
                    f'alter={state.relative_anchor.alter} '
                    f'octave={state.relative_anchor.octave}')

            elif isinstance(t, lyl.PitchCommand) and str(t) == r'\key':
                cursor.skip_while((lyl.Note, lyl.KeySignatureMode))

            elif isinstance(t, lyl.PitchCommand):
                pass   # other PitchCommand (e.g. \trill) -- silently skip

            elif isinstance(t, lyl.Tempo):
                cursor.skip_while((lyl.Length, lyl.EqualSign,
                                   lyl.IntegerValue))

            elif isinstance(t, lyl.SequentialEnd) and state.in_relative:
                state.in_relative = False
                state.relative_anchor = None

            elif isinstance(t, (lyl.Note, lyl.ChordStart)):
                cursor_s = self._emit_note(cursor, t, state, cursor_s)

            elif isinstance(t, lyl.Rest):
                length, dotted = cursor.read_length_dot()
                if length is not None:
                    state.current_length = length
                state.dotted = dotted
                dur_s = self._duration_s(state)
                dbg(f'rest @ {cursor_s:.3f}s: '
                    f'len={state.current_length} dotted={state.dotted} '
                    f'dur={dur_s:.3f}s')
                cursor_s += dur_s

            elif isinstance(t, lyl.Tie):
                state.tie_pending = True

            elif not isinstance(t, _IGNORABLE):
                wrn(f'skipping unsupported token '
                    f'{type(t).__name__} "{t}"')

        return cursor_s

    def parse(self, text: str) -> list[NoteEvent]:
        """
        Parse the specified LilyPond 'text' and return a flat list of notes
        with absolute start times, sorted by start time. Silence is implicit
        (absence of notes).
        """
        dbg(f'parse: {self._bpm} BPM, {self.spb:.4f}s per beat')

        skip = (ly.lex.Space, ly.lex.Comment)
        tokens = [t for t in ly.lex.state('lilypond').tokens(text)
                  if not isinstance(t, skip)]
        dbg(f'parse: {len(tokens)} tokens after strip')

        self._parse_tokens(TokenCursor(tokens), SequenceState())

        self.notes.sort(key=lambda n: n.start_s)
        dbg(f'parse: produced {len(self.notes)} note event(s)')
        return self.notes


def find_tempo(text: str) -> int | None:
    """
    Return BPM from the first \\tempo 4 = N directive, or None.
    """
    m = re.search(r'\\tempo\s+\d+\s*=\s*(\d+)', text)
    return int(m.group(1)) if m else None
