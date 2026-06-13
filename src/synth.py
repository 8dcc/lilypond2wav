from abc import ABC, abstractmethod

import numpy as np

from parser import NoteEvent


class Synthesizer(ABC):
    @abstractmethod
    def synthesize_note(self, freq: float, duration_s: float,
                        sample_rate: int) -> np.ndarray:
        """
        Return float32 PCM samples for a single pitched note.
        """
        ...

    def synthesize(self, notes: list[NoteEvent], sample_rate: int,
                   gate: float = 1.0) -> np.ndarray:
        """
        Return float32 PCM for all notes mixed into a single buffer.
        'gate' controls the fraction of each note's duration that sounds
        (0.0-1.0); the remainder is silence.
        """
        if not notes:
            return np.array([], dtype=np.float32)

        total_s = max(n.start_s + n.duration_s for n in notes)
        output = np.zeros(int(total_s * sample_rate), dtype=np.float32)

        for note in notes:
            chunk = self.synthesize_note(
                note.frequency,
                note.duration_s * gate,
                sample_rate
            )
            start = int(note.start_s * sample_rate)
            output[start:start + len(chunk)] += chunk

        return output


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


class HarmonicSynthesizer(Synthesizer):
    """
    Additive synthesizer: sums the first few harmonics of the
    fundamental with falling amplitudes and shapes each note with an
    ADSR envelope, giving a warmer tone than a plain sine wave.
    """

    # Relative amplitude of each harmonic, starting at the fundamental
    _HARMONIC_AMPS = (1.0, 0.5, 0.33, 0.25, 0.2, 0.16)

    _ATTACK_S = 0.010
    _DECAY_S = 0.080
    _SUSTAIN = 0.7
    _RELEASE_S = 0.060

    def synthesize_note(self, freq: float, duration_s: float,
                        sample_rate: int) -> np.ndarray:
        """
        Return float32 PCM samples for a single pitched note, built as
        an enveloped sum of harmonics of 'freq'.
        """
        n = int(duration_s * sample_rate)
        t = np.linspace(0.0, duration_s, n, endpoint=False)

        nyquist = sample_rate / 2.0
        wave = np.zeros(n, dtype=np.float64)
        total_amp = 0.0
        for k, amp in enumerate(self._HARMONIC_AMPS, start=1):
            harmonic_freq = freq * k
            if harmonic_freq >= nyquist:
                break
            wave += amp * np.sin(2.0 * np.pi * harmonic_freq * t)
            total_amp += amp

        if total_amp > 0.0:
            wave /= total_amp

        return wave.astype(np.float32) * self._adsr_envelope(n, sample_rate)

    def _adsr_envelope(self, n: int, sample_rate: int) -> np.ndarray:
        """
        Return a float32 attack/decay/sustain/release amplitude
        envelope of 'n' samples. Segment lengths are clamped so the
        envelope is valid even for very short notes.
        """
        attack_n = min(int(self._ATTACK_S * sample_rate), n)
        decay_n = min(int(self._DECAY_S * sample_rate), n - attack_n)
        release_n = min(int(self._RELEASE_S * sample_rate),
                        n - attack_n - decay_n)

        envelope = np.full(n, self._SUSTAIN, dtype=np.float32)
        if attack_n > 0:
            envelope[:attack_n] = np.linspace(0.0, 1.0, attack_n)
        if decay_n > 0:
            envelope[attack_n:attack_n + decay_n] = \
                np.linspace(1.0, self._SUSTAIN, decay_n)
        if release_n > 0:
            envelope[n - release_n:] = \
                np.linspace(self._SUSTAIN, 0.0, release_n)

        return envelope
