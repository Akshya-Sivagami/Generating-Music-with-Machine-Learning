"""Tempo-aware MIDI reading, measured metrics, and paired-note MIDI writing."""
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
import math

import mido
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SEED = 229
WINDOW_FEATURES = ["mean_pitch", "std_pitch", "pitch_range", "mean_duration",
                   "std_duration", "mean_onset_chord_size"]
CONTEXT_FEATURES = ["previous_mean_pitch", "previous_std_pitch",
                    "previous_pitch_range", "previous_chord_size", "previous_mean_duration"]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: int
    velocity: int = 80


def read_midi(path):
    """Merge synchronous tracks; apply every tempo change; pair by channel/pitch.

    Pedal effects are excluded: durations measure key-down time, not acoustic sustain.
    Overlapping same-pitch strikes use FIFO pairing. Malformed dangling notes fail.
    """
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError(f"Unsupported asynchronous or SMPTE MIDI: {path}")
    active = defaultdict(deque)
    notes = []
    now = 0.0
    for msg in midi:  # Mido converts merged delta ticks to seconds with the tempo map.
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)].append((now, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            queue = active[(msg.channel, msg.note)]
            if queue:
                start, velocity = queue.popleft()
                if now > start:
                    notes.append(Note(start, now, msg.note, velocity))
    if any(active.values()):
        raise ValueError(f"Unclosed note-on events: {path}")
    return sorted(notes, key=lambda n: (n.start, n.pitch)), now


def group_onsets(notes, tolerance=0.03):
    """Group performed chord strikes within 30 ms of the first strike, never chain."""
    groups = []
    for note in notes:
        if not groups or note.start - groups[-1][0].start > tolerance:
            groups.append([note])
        else:
            groups[-1].append(note)
    return groups


def context_features(group):
    pitches = np.array([n.pitch for n in group], dtype=float)
    durations = np.array([n.end - n.start for n in group], dtype=float)
    return np.array([pitches.mean(), pitches.std(), np.ptp(pitches),
                     len(group), durations.mean()])


def controller_features(context, mode):
    """Mode plus context and mode-context interactions for conditional ridge regression."""
    return np.r_[float(mode), context, float(mode) * context]


def window_features(notes):
    pitches = np.array([n.pitch for n in notes], dtype=float)
    durations = np.array([n.end - n.start for n in notes], dtype=float)
    groups = group_onsets(notes)
    return [pitches.mean(), pitches.std(), np.ptp(pitches), durations.mean(),
            durations.std(), len(notes) / len(groups)]


def midi_metrics(path):
    notes, duration = read_midi(path)
    starts = np.array([n.start for n in notes])
    distinct = np.unique(starts)
    return {
        "note_count": len(notes), "duration_seconds": duration,
        "notes_per_second": len(notes) / duration if duration else 0.0,
        "mean_inter_onset_seconds": float(np.diff(distinct).mean()) if len(distinct) > 1 else 0.0,
        "unique_pitches": len({n.pitch for n in notes}),
    }


def write_midi(path, notes, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be positive and finite")
    midi = mido.MidiFile(type=0, ticks_per_beat=960)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    tempo = 500000
    track.extend([mido.MetaMessage("set_tempo", tempo=tempo),
                  mido.Message("program_change", program=0)])
    final_tick = round(mido.second2tick(duration, midi.ticks_per_beat, tempo))
    events = []
    for n in notes:
        if not (0 <= n.start < n.end <= duration and 0 <= n.pitch <= 127):
            raise ValueError(f"Invalid generated note: {n}")
        start = round(mido.second2tick(n.start, midi.ticks_per_beat, tempo))
        end = min(final_tick, max(start + 1, round(mido.second2tick(n.end, midi.ticks_per_beat, tempo))))
        if start >= final_tick:
            continue
        events.append((start, 1, mido.Message("note_on", note=n.pitch, velocity=n.velocity)))
        events.append((end, 0, mido.Message("note_off", note=n.pitch, velocity=0)))
    previous = 0
    for tick, _, msg in sorted(events, key=lambda x: (x[0], x[1], x[2].note)):
        track.append(msg.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=final_tick - previous))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
