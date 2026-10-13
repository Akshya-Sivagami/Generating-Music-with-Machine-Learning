"""Read generated files, compare paired seeds, and draw reproducible result plots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extension.common import ROOT, midi_metrics, read_midi, save_json


def evaluate(output_dir=ROOT / "outputs"):
    output_dir = Path(output_dir)
    rows = []
    settings = {}
    for mode in ["sparse", "dense", "uncontrolled"]:
        paths = sorted((output_dir / "trials").glob(f"{mode}_*.mid"))
        if not paths and (output_dir / f"{mode}_generated.mid").exists():
            paths = [output_dir / f"{mode}_generated.mid"]
        for path in paths:
            metadata = json.loads(path.with_suffix(".json").read_text())
            settings[(metadata["seed"], mode)] = {k: metadata[k] for k in
                ["target_seconds", "baseline", "seed_midi", "controller_sha256", "sampling"]}
            rows.append({"mode": mode, "seed": metadata["seed"],
                         "file": path.relative_to(output_dir).as_posix(), **midi_metrics(path)})
    if not rows or not {"sparse", "dense"} <= {r["mode"] for r in rows}:
        raise ValueError("Generate both sparse and dense MIDI before evaluation")
    paired = {}
    for row in rows:
        paired.setdefault(row["seed"], {})[row["mode"]] = row
    for seed, modes in paired.items():
        if not {"sparse", "dense"} <= set(modes):
            raise ValueError(f"Unpaired seed {seed}")
        if abs(modes["sparse"]["duration_seconds"] - modes["dense"]["duration_seconds"]) > 0.001:
            raise ValueError(f"Mismatched duration for seed {seed}")
        if settings[(seed, "sparse")] != settings[(seed, "dense")]:
            raise ValueError(f"Mismatched generator, controller, seed context or sampling settings for seed {seed}")
        if any(m["unique_pitches"] < 2 for m in modes.values()):
            raise ValueError(f"Insufficient note variation for seed {seed}")
    differences = [m["dense"]["notes_per_second"] - m["sparse"]["notes_per_second"] for m in paired.values()]
    summary = {}
    for mode in ["sparse", "dense", "uncontrolled"]:
        values = np.array([r["notes_per_second"] for r in rows if r["mode"] == mode])
        if len(values):
            summary[mode] = {"trials": len(values), "mean_notes_per_second": float(values.mean()),
                             "std_notes_per_second": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    report = {"summary": summary, "paired_seeds": sorted(paired),
              "dense_exceeds_sparse_trials": int(sum(d > 0 for d in differences)),
              "paired_trials": len(differences), "mean_paired_difference": float(np.mean(differences)),
              "all_pairs_pass": all(d > 0 for d in differences), "measurements": rows,
              "mean_inter_onset_definition": "gap between exact distinct onset times; simultaneous notes count separately in density"}
    stats_path = ROOT / "datasets/statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        controlled = [r for r in rows if r["mode"] in ["sparse", "dense"]]
        report["requested_training_regime_passes"] = sum(
            r["notes_per_second"] <= stats["sparse_max_notes_per_second"] if r["mode"] == "sparse"
            else r["notes_per_second"] >= stats["dense_min_notes_per_second"] for r in controlled)
        report["controlled_trials"] = len(controlled)
    save_json(output_dir / "density_comparison.json", report)
    with (output_dir / "density_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    modes = [m for m in ["sparse", "dense", "uncontrolled"] if m in summary]
    means = [summary[m]["mean_notes_per_second"] for m in modes]
    stds = [summary[m]["std_notes_per_second"] for m in modes]
    ax.bar([m.title() for m in modes], means, yerr=stds, capsize=6,
           color=["#357B87", "#CB6847", "#899298"][:len(modes)], width=0.55)
    for i, value in enumerate(means):
        ax.text(i, value + stds[i] + 0.2, f"{value:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.24)
    ax.set_ylabel("Note-on events per second")
    ax.set_title("Learned note density during generation")
    ax.text(0.5, -0.17, f"{len(paired)} matched seeds | same checkpoint, input and duration | bars: mean +/- sample SD",
            ha="center", va="top", transform=ax.transAxes, fontsize=9)
    fig.savefig(plots / "density_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, mode, color in zip(axes, ["sparse", "dense"], ["#357B87", "#CB6847"]):
        path = output_dir / f"{mode}_generated.mid"
        if not path.exists():
            path = output_dir / next(r["file"] for r in rows if r["mode"] == mode)
        notes, duration = read_midi(path)
        for n in notes:
            ax.plot([n.start, n.end], [n.pitch, n.pitch], lw=2, color=color)
        metrics = midi_metrics(path)
        ax.set_title(f"{mode.title()}: {metrics['notes_per_second']:.2f} notes/sec, {len(notes)} notes", loc="left")
        ax.set_ylabel("MIDI pitch")
        ax.set_xlim(0, duration)
    axes[-1].set_xlabel("Time (seconds)")
    fig.savefig(plots / "piano_roll.png", dpi=180)
    plt.close(fig)
    dataset = ROOT / "datasets/density.npz"
    if dataset.exists():
        with np.load(dataset, allow_pickle=False) as data:
            train_density = data["density"][data["split"] == "train"]
        stats = json.loads((ROOT / "datasets/statistics.json").read_text())
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.hist(train_density, bins=30, color="#899298", edgecolor="white")
        for key, label, color in [("sparse_max_notes_per_second", "Sparse maximum", "#357B87"),
                                   ("dense_min_notes_per_second", "Dense minimum", "#CB6847")]:
            ax.axvline(stats[key], label=f"{label}: {stats[key]:.2f}", color=color, lw=2)
        ax.set_xlabel("Training-window note-on events per second")
        ax.set_ylabel("Four-second windows")
        ax.set_title("Density labels derived from training pieces only")
        ax.legend()
        fig.savefig(plots / "training_distribution.png", dpi=180)
        plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "measurements"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    evaluate(parser.parse_args().output_dir)
