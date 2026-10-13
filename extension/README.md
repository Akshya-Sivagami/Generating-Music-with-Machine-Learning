# Controllable Music Generation using Learned Note Density

This extension adds a learned sparse/dense controller to the existing encoder-decoder.
It changes note emission decisions **inside autoregressive decoding**. The baseline
provides pitch probabilities; a trained conditional model predicts onset spacing,
note duration and chord size. Every chosen event is then fed back to the decoder.
There is no generation-then-delete/duplicate stage.

## Setup and commands

Use Python 3.12 from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r extension/requirements.txt
.\.venv\Scripts\python.exe -m extension.download_assets
.\.venv\Scripts\python.exe -m extension.run_demo --render-audio
```

The small trained models are versioned. For a complete retraining run:

```powershell
.\.venv\Scripts\python.exe -m extension.run_demo --rebuild --trials 5 --seconds 30 --seed 229 --render-audio
```

Individual commands, using the activated virtual environment or its Python executable:

```text
python -m extension.extract_density_dataset
python -m extension.train_density_model
python -m extension.generate_density_controlled --mode sparse
python -m extension.generate_density_controlled --mode dense
python -m extension.evaluate_density --output-dir extension/outputs/runs/seed_229_n_5_seconds_30
python -m unittest discover -s extension/tests -v
```

Use `.venv/bin/python` on macOS/Linux. `requirements-lock.txt` records the versions of
the successful local experiment. CPU inference is sufficient; no CUDA is required.

MIDI files: `outputs/sparse_generated.mid` and `outputs/dense_generated.mid`. WAV
previews, if requested, have the same names and are generated locally. The WAV
synthesizer uses simple additive tones, so it demonstrates texture/timing rather
than concert-piano timbre. WAV files are ignored by Git and can be recreated.

## Data, labels and leakage controls

The downloaded source is **MAESTRO v3.0.0**, a substitute performance corpus rather
than the original paper's processed classical dataset. `download_assets` selects 28
Chopin performances: 16 train, 6 validation, 6 test. It preserves MAESTRO's official
composition-disjoint splits, sorts titles by SHA256 of `229:title`, and keeps one
performance per exact title. The manifest lists original filenames, titles, split,
license and per-file SHA256 hashes. Rebuild selection uses seed 229 independently
of the optional model/generation seed.

Only the first 240 seconds of each performance are used. Windows are four seconds,
non-overlapping, anchored to the first note. Incomplete final windows and windows
with fewer than two notes are omitted. Note density is **positive-velocity note-on
events / four seconds**; every pitch in a chord counts separately. Track merging and
the complete tempo map are applied, so variable MIDI tempo is handled in seconds.
Pedal extension is excluded; duration means key-down time.

The 40th and 60th percentiles are calculated from training windows only:

- Sparse: density <= 7.25 notes/sec.
- Dense: density >= 11.00 notes/sec.
- Middle windows: excluded from classifier/controller fitting and evaluation.

The resulting 1,540 windows include 883 train, 323 validation and 334 test windows.
After excluding the middle band, there are 714, 270 and 255 classified windows.
Thresholds and normalizers are never fitted on validation/test data. Neighboring
windows from the same performance remain in one split.

## Two small learned models, with different roles

**Diagnostic classifier:** mean pitch, pitch standard deviation, pitch range, mean
duration, duration standard deviation and mean onset chord size predict sparse/dense.
Note count, current density and current inter-onset gaps are deliberately excluded
from inputs. Duration/polyphony are still correlated with the automatic labels;
classification measures label recoverability, not subjective musical quality.

Compare a majority baseline, logistic regression and a 16/8-hidden-unit MLP. Select
between learned classifiers by validation **balanced accuracy**; then report test
metrics. Logistic regression was selected, even though the MLP has higher raw test
accuracy: test results are not used for model selection.

**Generative controller:** a multi-output ridge regressor learns log onset gap,
log mean note duration and log chord size. Inputs are requested mode, previous
onset-group mean pitch, pitch standard deviation, pitch span, note count, mean
duration, and mode-context interactions. Onset groups contain performed strikes
within 30 ms of the first strike; grouping never chains successive notes indefinitely.

The previous onset is the causal context. Targets describe the current onset. Its
training-window label supplies the requested density regime. Scaling and ridge
coefficients are fitted on training events only. All outputs are constrained to the
training set's global 1st/99th percentiles, identically for both modes. These bounds
are validity constraints, not hand-selected sparse/dense targets.

The classifier is diagnostic; **the conditional ridge model is the learned component
actually used for generation**. This distinction matters when explaining the feature.

## Integration with the completed baseline

`encoder_adapter.py` imports the existing `EncoderRNN` and `DecoderRNN` classes
without executing training. It loads the original epoch-105 weights with
`torch.load(weights_only=True)` and validates the checkpoint SHA256. All baseline
source files and baseline outputs stay unchanged.

Seed MIDI defaults to the bundled `202.mid`. The adapter chordifies its first 100
note/rest segments into 219-dimensional vectors; a segment represents active pitches
and duration. Seconds are normalized to the baseline's 120 BPM duration vocabulary.
Time/key defaults are 4/4 and zero sharps. These are documented seed-adapter
approximations. Start/end vectors follow the existing generation script's 2/3 values.

At each decoding step:

1. Run the frozen decoder to obtain pitch/duration probabilities.
2. Predict spacing, duration and chord size using requested mode and previous context.
3. Sample the predicted number of distinct pitches from the baseline's highest-ranked
   piano candidates (range 21-108, at least 12 candidates, temperature 0.65).
4. Advance the onset clock, emit the selected chord, and feed its vector back.

All sampling parameters are the same for sparse and dense. Seed, source MIDI,
checkpoint and 30-second target duration are matched. Retriggers end any earlier
same-pitch note rather than leaving stuck notes. Output ends at a fixed MIDI
end-of-track time, including trailing silence. Metrics are recomputed by reading
the saved MIDI files.

`--include-uncontrolled` adds a reference sampler with controller-disabled timing,
duration argmax and the baseline 0.5 pitch threshold. That sampler uses the extension
adapter and discretized feedback. It is **not** a re-evaluation of the original
two-pass generation script or its saved music, and its deterministic five trials
have no seed variation. It is a descriptive reference, not an isolated causal
ablation of every sampling change.

## Results and artifacts

See `report.md` and the machine-readable JSON/CSV for measured results. Five matched
trials produce 5.53 +/- 0.24 sparse versus 12.41 +/- 0.05 dense notes/sec (sample SD).
Dense exceeds sparse for every pair, and each mode's global mean is within its
learned training density regime.

- `datasets/manifest.json`: data provenance and split assignment.
- `datasets/statistics.json`: window counts, features and training-only thresholds.
- `models/`: two tiny trained models, under 2 KB each in this experiment.
- `outputs/model_evaluation.json`: all classifier/controller comparisons.
- `outputs/density_comparison.json` and `.csv`: measurements read from generated MIDI.
- `outputs/runs/`: MIDI and compact provenance for each trial. Per-step decoder
  traces (`*.trace.jsonl`) are produced locally and ignored by Git.
- `plots/`: density comparison, shared-scale piano rolls and training distribution.

Load the included joblib models only from this trusted project; retrain to recreate
them. They require compatible scikit-learn versions (see the lock file). The large
source checkpoint, archive, extracted MIDI, arrays, Python environment and caches
are ignored by Git. Downloads use verified HTTPS and checksums; if a source link
changes, supply a matching verified checkpoint locally rather than substituting
random weights or claiming an unrun result.

## Limitations and remaining review

This is a compact conditional event model with a frozen baseline pitch generator,
not a retrained end-to-end density-conditioned RNN. Log-target regression produces
smooth typical values, not the full distribution of expressive timing. The data
subset and seed input are limited. Classification samples from one piece are
correlated; the composition-disjoint test split improves evaluation but does not
make individual windows independent. Labels describe density, not complexity or
quality. Reported results cover one seed context and five random seeds, not universal
control on all music.

MIDI parsing, paired note lifetimes, changing pitches, density ordering and exact-seed
reproducibility have been verified. Human listening and teammate approval are still
required before a classroom-quality claim or merge into `main`.

## Attribution

David Kang, Jung Youn Kim and Simen Ringdahl,
[Generating music with Machine Learning, Stanford CS229 (2018)](https://cs229.stanford.edu/proj2018/report/18.pdf).
Checkpoint comes from the paper's authors' public Dropbox code release; it is not
redistributed in Git. The repository does not specify a blanket license for the
authors' code/checkpoint; their attribution is retained.

Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
Sander Dieleman, Erich Elsen, Jesse Engel and Douglas Eck. *Enabling Factorized Piano
Music Modeling and Generation with the MAESTRO Dataset*, ICLR 2019.
[MAESTRO v3.0.0 dataset](https://magenta.tensorflow.org/datasets/maestro), Google LLC,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
