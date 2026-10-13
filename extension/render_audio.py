"""Portable listening previews synthesized from actual generated MIDI note events."""
import argparse
from pathlib import Path
import wave

import numpy as np
from extension.common import read_midi


def render(path, output=None, sample_rate=22050):
    notes, duration = read_midi(path)
    signal = np.zeros(round(duration * sample_rate), dtype=np.float64)
    for note in notes:
        start = round(note.start * sample_rate)
        count = min(len(signal) - start, max(1, round((note.end - note.start) * sample_rate)))
        t = np.arange(count) / sample_rate
        frequency = 440 * 2 ** ((note.pitch - 69) / 12)
        envelope = np.minimum(t / 0.008, 1) * np.exp(-2.5 * t)
        envelope *= np.minimum((count / sample_rate - t) / 0.03, 1)
        tone = sum(weight * np.sin(2 * np.pi * frequency * harmonic * t)
                   for harmonic, weight in [(1, 1.0), (2, 0.35), (3, 0.16), (4, 0.07)]
                   if frequency * harmonic < sample_rate / 2)
        signal[start:start + count] += note.velocity / 127 * envelope * tone
    peak = np.max(abs(signal))
    if peak == 0:
        raise ValueError("MIDI contains no audible notes")
    pcm = np.asarray(signal / peak * 0.85 * 32767, dtype="<i2")
    output = Path(output) if output else Path(path).with_suffix(".wav")
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    print(f"Synthesized listening preview: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    render(args.midi, args.output)
