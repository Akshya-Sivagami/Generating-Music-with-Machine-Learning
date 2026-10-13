"""Learned timing/polyphony control inside the pretrained encoder-decoder loop."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from extension.common import ROOT, REPO, SEED, Note, context_features, controller_features, read_midi, group_onsets, write_midi, midi_metrics, save_json
from extension.encoder_adapter import load_baseline, seed_sequence, encode_event, durations

DEFAULT_SEED_MIDI = REPO / "original_replication/ml_code_simen/ReadMidi/Midi/202.mid"


def generate(mode, output, seconds=30.0, seed=SEED, seed_midi=DEFAULT_SEED_MIDI, loaded=None):
    if mode not in ("sparse", "dense", "uncontrolled"):
        raise ValueError("Mode must be sparse, dense or uncontrolled")
    if not np.isfinite(seconds) or not 1 <= seconds <= 300:
        raise ValueError("Duration must be between 1 and 300 seconds")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    encoder, decoder, provenance = load_baseline() if loaded is None else loaded
    controller_path = ROOT / "models/density_controller.joblib"
    if not controller_path.exists():
        raise FileNotFoundError("Missing controller. Run python -m extension.train_density_model")
    control = joblib.load(controller_path)
    input_notes, _ = read_midi(seed_midi)
    context = context_features(group_onsets(input_notes)[0])
    src = seed_sequence(seed_midi)
    notes, trace = [], []
    now = 0.0
    pitch_range = np.arange(21, 109)
    vocabulary = durations()
    with torch.inference_mode():
        hidden, cell = encoder(src.view(102, 1, 219))
        incoming = src[0]
        for step in range(20000):
            probability, hidden, cell = decoder(incoming.view(1, 1, 219), hidden, cell)
            vector = probability.view(-1).cpu().numpy()
            if mode == "uncontrolled":
                gap = float(vocabulary[np.argmax(vector[128:192])] / 2)
                duration = gap
                pitches = pitch_range[vector[pitch_range] > 0.5].tolist()
            else:
                features = controller_features(context, int(mode == "dense"))
                predicted = np.exp(control["model"].predict(features[None])[0])
                gap, duration, size = np.clip(predicted, control["bounds"][0], control["bounds"][1])
                size = int(np.clip(round(size), 1, 88))
                # Pitch candidates come from the frozen baseline; no MIDI exists to edit here.
                candidates = pitch_range[np.argsort(vector[pitch_range])[-max(12, size):]]
                weights = np.maximum(vector[candidates], 1e-8) ** (1 / 0.65)
                pitches = sorted(rng.choice(candidates, size=size, replace=False, p=weights / weights.sum()).tolist())
            if now + gap >= seconds:
                break
            now += float(gap)
            end = min(seconds, now + float(duration))
            # Retriggers end an earlier same-pitch note. Every selected onset is preserved.
            for i in range(len(notes) - 1, -1, -1):
                n = notes[i]
                if n.pitch in pitches and n.end > now:
                    notes[i] = Note(n.start, now, n.pitch, n.velocity)
            group = [Note(now, end, int(p)) for p in pitches]
            notes.extend(group)
            trace.append({"step": step, "time": now, "gap": float(gap),
                          "duration": float(duration), "pitches": pitches})
            if group:
                context = context_features(group)
            # Chosen event feeds back into the next decoder input; timing is controlled before emission.
            incoming = torch.from_numpy(encode_event(pitches, duration))
        else:
            raise RuntimeError("Decoder exceeded safety step limit")
    if len({n.pitch for n in notes}) < 2:
        raise RuntimeError("Generation produced insufficient pitch variation; inspect checkpoint/seed")
    write_midi(output, notes, seconds)
    metrics = midi_metrics(output)  # Reopen the actual file rather than trusting in-memory counts.
    report = {"mode": mode, "seed": seed, "target_seconds": seconds, "metrics": metrics,
              "baseline": provenance, "seed_midi": str(Path(seed_midi).relative_to(REPO)) if Path(seed_midi).is_relative_to(REPO) else str(seed_midi),
              "controller_sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
              "sampling": {"temperature": 0.65, "candidate_pool_minimum": 12, "piano_pitch_range": [21, 108]},
              "event_count": len(trace), "trace_file": Path(output).with_suffix(".trace.jsonl").name}
    Path(output).with_suffix(".trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in trace), encoding="utf-8")
    save_json(Path(output).with_suffix(".json"), report)
    print(f"{mode.upper()}: {metrics['note_count']} notes, {metrics['duration_seconds']:.2f}s, "
          f"{metrics['notes_per_second']:.3f} notes/sec -> {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sparse", "dense", "uncontrolled"], required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seed-midi", type=Path, default=DEFAULT_SEED_MIDI)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / f"{args.mode}_generated.mid"
    generate(args.mode, output, args.seconds, args.seed, args.seed_midi)


if __name__ == "__main__":
    main()
