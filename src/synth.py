from abc import ABC, abstractmethod

import numpy as np

from parser import Event, NoteEvent, RestEvent


class Synthesizer(ABC):
    @abstractmethod
    def synthesize_note(self, freq: float, duration_s: float,
                        sample_rate: int) -> np.ndarray:
        """
        Return float32 PCM samples for a single pitched note.
        """
        ...

    def synthesize(self, events: list[Event], sample_rate: int,
                   gate: float = 1.0) -> np.ndarray:
        """
        Return concatenated float32 PCM for all events. 'gate' controls
        what fraction of each note's duration is sounded (0.0-1.0); the
        remainder is silence.
        """
        chunks = []
        for event in events:
            if isinstance(event, NoteEvent):
                note_dur = event.duration * gate
                rest_dur = event.duration * (1.0 - gate)
                chunks.append(self.synthesize_note(
                    event.frequency, note_dur, sample_rate))
                if rest_dur > 0.0:
                    chunks.append(np.zeros(
                        int(rest_dur * sample_rate), dtype=np.float32))
            else:
                n = int(event.duration * sample_rate)
                chunks.append(np.zeros(n, dtype=np.float32))

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)


class SineSynthesizer(Synthesizer):
    _ATTACK_S = 0.005   # 5 ms linear attack
    _RELEASE_S = 0.010  # 10 ms linear release

    def synthesize_note(self, freq: float, duration_s: float,
                        sample_rate: int) -> np.ndarray:
        n = int(duration_s * sample_rate)
        t = np.linspace(0.0, duration_s, n, endpoint=False)
        wave = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

        attack_n = min(int(self._ATTACK_S * sample_rate), n)
        release_n = min(int(self._RELEASE_S * sample_rate), n - attack_n)

        envelope = np.ones(n, dtype=np.float32)
        if attack_n > 0:
            envelope[:attack_n] = np.linspace(0.0, 1.0, attack_n)
        if release_n > 0:
            envelope[n - release_n:] = np.linspace(1.0, 0.0, release_n)

        return wave * envelope
