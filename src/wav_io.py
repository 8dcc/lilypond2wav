import wave

import numpy as np

SAMPLE_RATE = 44100


def write_wav(samples: np.ndarray, path: str,
              sample_rate: int = SAMPLE_RATE) -> None:
    """
    Write float32 samples as 16-bit mono WAV to the specified 'path'.
    Normalizes to the full int16 range before writing.
    """
    peak = np.max(np.abs(samples))
    if peak > 0:
        normalized = samples / peak
    else:
        normalized = samples
    int_samples = (normalized * 32767).astype(np.int16)

    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
