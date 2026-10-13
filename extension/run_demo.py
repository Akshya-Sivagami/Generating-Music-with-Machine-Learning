"""One command to rebuild or run the learned sparse/dense generation demo."""
import argparse
import shutil

from extension.common import ROOT, SEED
from extension.encoder_adapter import load_baseline
from extension.generate_density_controlled import generate
from extension.evaluate_density import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Download assets, extract windows, and train controller")
    parser.add_argument("--mode", choices=["both", "sparse", "dense", "uncontrolled"], default="both")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--include-uncontrolled", action="store_true")
    parser.add_argument("--render-audio", action="store_true", help="Create portable WAV listening previews")
    args = parser.parse_args()
    if not 1 <= args.trials <= 100:
        parser.error("trials must be between 1 and 100")
    if args.rebuild:
        from extension.download_assets import prepare_assets
        from extension.extract_density_dataset import extract
        from extension.train_density_model import train
        prepare_assets()
        extract()
        train(args.seed)
    loaded = load_baseline()
    if args.mode != "both":
        path = ROOT / "outputs" / f"{args.mode}_generated.mid"
        generate(args.mode, path, args.seconds, args.seed, loaded=loaded)
        if args.render_audio:
            from extension.render_audio import render
            render(path)
        return
    # A unique run folder avoids accidentally counting trials from an older experiment.
    run = ROOT / "outputs" / "runs" / f"seed_{args.seed}_n_{args.trials}_seconds_{args.seconds:g}"
    run.mkdir(parents=True, exist_ok=True)
    trials = run / "trials"
    trials.mkdir(exist_ok=True)
    modes = ["sparse", "dense"] + (["uncontrolled"] if args.include_uncontrolled else [])
    expected = {f"{m}_{args.seed+i}.mid" for m in modes for i in range(args.trials)}
    # Only remove old generated trial files inside this known run's trials directory.
    for path in trials.glob("*.mid"):
        if path.name not in expected:
            path.unlink()
            path.with_suffix(".json").unlink(missing_ok=True)
    for i in range(args.trials):
        for mode in modes:
            path = trials / f"{mode}_{args.seed+i}.mid"
            metadata = generate(mode, path, args.seconds, args.seed + i, loaded=loaded)
            if i == 0:
                for suffix in [".mid", ".json"]:
                    shutil.copyfile(path.with_suffix(suffix), ROOT / "outputs" / f"{mode}_generated{suffix}")
                    shutil.copyfile(path.with_suffix(suffix), run / f"{mode}_generated{suffix}")
                for destination in [ROOT / "outputs", run]:
                    shutil.copyfile(path.with_suffix(".trace.jsonl"), destination / metadata["trace_file"])
    report = evaluate(run)
    for name in ["density_comparison.json", "density_comparison.csv"]:
        shutil.copyfile(run / name, ROOT / "outputs" / name)
    if not report["all_pairs_pass"]:
        raise SystemExit("Control comparison failed; see actual measurements. No successful result claimed.")
    if args.render_audio:
        from extension.render_audio import render
        for mode in modes:
            render(ROOT / "outputs" / f"{mode}_generated.mid")


if __name__ == "__main__":
    main()
