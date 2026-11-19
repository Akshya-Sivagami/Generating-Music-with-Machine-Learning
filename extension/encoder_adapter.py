"""Import the unchanged baseline model classes; adapt MIDI to their 219-vector interface."""
import ast
import hashlib
import importlib.util
from functools import lru_cache

import numpy as np
import torch

from extension.common import ROOT, REPO, read_midi
from extension.download_assets import CHECKPOINT_SHA


@lru_cache(maxsize=1)
def durations():
    source = REPO / "original_replication" / "encoder_decoder" / "EncDec_music_creation.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "note_duration" for t in statement.targets):
            values = ast.literal_eval(statement.value)
            # Original 64th duration class is an overflow bucket (>10 beats).
            return np.array(values + [10.0], dtype=float)
    raise ValueError("Cannot find baseline duration vocabulary")


def encode_event(pitches, duration_seconds):
    result = np.zeros(219, dtype=np.float32)
    result[list(pitches)] = 1
    result[128 + int(np.argmin(abs(durations() - duration_seconds * 2)))] = 1
    result[194] = 1  # 4/4 time
    result[201] = 1
    result[211] = 1  # C major / A minor
    return result


def seed_sequence(path):
    """Chordify all note-on/off boundaries for the first 100 segments at real MIDI tempo.

    Normalize seconds to the baseline's fixed 120 BPM duration vocabulary; include
    rests. Time/key metadata defaults to 4/4 and zero sharps for this extension demo.
    """
    notes, _ = read_midi(path)
    if not notes:
        raise ValueError("Seed MIDI contains no notes")
    boundaries = sorted({v for n in notes for v in (n.start, n.end)})
    rows = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < 0.001:
            continue
        pitches = sorted({n.pitch for n in notes if n.start <= start + 1e-8 and n.end > start + 1e-8})
        rows.append(encode_event(pitches, end - start))
        if len(rows) == 100:
            break
    if len(rows) < 100:
        raise ValueError("Seed must have at least 100 chordified note/rest segments")
    # Retain the baseline generation script's start/end vectors (2/3).
    return torch.from_numpy(np.vstack([np.full(219, 2, dtype=np.float32), rows,
                                       np.full(219, 3, dtype=np.float32)]))


def load_baseline(checkpoint=None):
    path = ROOT / "assets" / "baseline_checkpoint.pth" if checkpoint is None else checkpoint
    if not path.exists():
        raise FileNotFoundError("Missing pretrained checkpoint. Run python -m extension.download_assets")
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != CHECKPOINT_SHA:
        raise ValueError("Checkpoint differs from the verified original epoch-105 asset")
    source = REPO / "original_replication" / "encoder_decoder" / "encoderdecoder_model.py"
    spec = importlib.util.spec_from_file_location("baseline_encoderdecoder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder = module.EncoderRNN(219, 512)
    decoder = module.DecoderRNN(512, 219)
    encoder.load_state_dict(state["enc_state_dict"])
    decoder.load_state_dict(state["dec_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, {"epoch": int(state["epoch"]), "sha256": sha}
