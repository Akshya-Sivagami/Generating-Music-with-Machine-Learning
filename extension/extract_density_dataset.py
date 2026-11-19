"""Extract non-overlapping windows and causal event contexts from MIDI."""
import argparse
import json
from collections import Counter

import numpy as np
from extension.common import ROOT, WINDOW_FEATURES, CONTEXT_FEATURES, read_midi, group_onsets, context_features, window_features, save_json


def extract(window_seconds=4.0, max_seconds=240.0):
    manifest = json.loads((ROOT / "datasets" / "manifest.json").read_text())
    windows, events = [], []
    for piece, row in enumerate(manifest["files"]):
        notes, duration = read_midi(ROOT / "assets" / "midi" / row["local_path"])
        if len(notes) < 2:
            raise ValueError(f"Not enough notes: {row['local_path']}")
        origin = notes[0].start
        end = min(duration, origin + max_seconds)
        count = int((end - origin) // window_seconds)
        piece_windows = {}
        for index in range(count):
            start = origin + index * window_seconds
            selected = [n for n in notes if start <= n.start < start + window_seconds]
            if len(selected) < 2:
                continue  # Empty/near-empty windows cannot provide stable pitch features.
            piece_windows[index] = len(windows)
            windows.append({"piece": piece, "split": row["split"], "index": index,
                            "start": start, "density": len(selected) / window_seconds,
                            "features": window_features(selected)})
        groups = group_onsets([n for n in notes if n.start < origin + count * window_seconds])
        for previous, current in zip(groups, groups[1:]):
            index = int((current[0].start - origin) // window_seconds)
            if index not in piece_windows:
                continue
            # Input is previous onset context; outputs are current onset gap, duration, size.
            target = [current[0].start - previous[0].start,
                      np.mean([n.end - n.start for n in current]), len(current)]
            events.append({"window": piece_windows[index], "context": context_features(previous),
                           "target": target})
    if not windows or not events:
        raise ValueError("No usable training windows")
    train_density = [w["density"] for w in windows if w["split"] == "train"]
    lower, upper = np.quantile(train_density, [0.4, 0.6])
    if lower >= upper:
        raise ValueError("Training densities have no distinct 40/60 quantiles; add more varied MIDI")
    labels = np.array([0 if w["density"] <= lower else 1 if w["density"] >= upper else -1 for w in windows])
    arrays = {
        "X": np.array([w["features"] for w in windows]), "y": labels,
        "density": np.array([w["density"] for w in windows]),
        "split": np.array([w["split"] for w in windows]),
        "piece": np.array([w["piece"] for w in windows]),
        "event_window": np.array([e["window"] for e in events]),
        "event_context": np.array([e["context"] for e in events]),
        "event_target": np.array([e["target"] for e in events]),
    }
    dataset = ROOT / "datasets"
    dataset.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset / "density.npz", **arrays)
    stats = {
        "window_seconds": window_seconds, "max_seconds_per_piece": max_seconds,
        "onset_group_tolerance_seconds": 0.03,
        "density_definition": "positive-velocity note-on events / full window duration in seconds",
        "threshold_source": "Training split only", "sparse_max_notes_per_second": float(lower),
        "dense_min_notes_per_second": float(upper), "window_features": WINDOW_FEATURES,
        "controller_context_features": CONTEXT_FEATURES,
        "controller_targets": ["onset_gap_seconds", "note_duration_seconds", "chord_size"],
        "windows_total": len(windows), "events_total": len(events),
        "splits": {s: {"pieces": len(set(arrays["piece"][arrays["split"] == s].tolist())),
                       "labels": dict(Counter(str(int(y)) for y in labels[arrays["split"] == s]))}
                   for s in ["train", "validation", "test"]},
    }
    save_json(dataset / "statistics.json", stats)
    print(json.dumps(stats, indent=2))
    return arrays, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-seconds", type=float, default=4)
    parser.add_argument("--max-seconds", type=float, default=240)
    args = parser.parse_args()
    if not (args.window_seconds > 0 and args.max_seconds >= args.window_seconds):
        parser.error("Use a positive window and max-seconds at least one window")
    extract(args.window_seconds, args.max_seconds)


if __name__ == "__main__":
    main()
