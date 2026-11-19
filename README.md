# Generating Music with Machine Learning

Replication of the four approaches in David Kang, Jung Youn Kim and Simen Ringdahl's
[Stanford CS229 project](https://cs229.stanford.edu/proj2018/report/18.pdf), with a new
**Controllable Music Generation using Learned Note Density** extension.

The completed replication is under `original_replication/`; its generated MIDI and
recorded evaluations are under `results/`. The extension imports the existing
encoder-decoder classes and reuses the authors' epoch-105 checkpoint without
retraining or changing the four baseline implementations.

## Run the sparse/dense demo

Use Python 3.12. Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r extension/requirements.txt
.\.venv\Scripts\python.exe -m extension.download_assets
.\.venv\Scripts\python.exe -m extension.run_demo --render-audio
```

Small trained controller/classifier artifacts and measured demo results are included.
The original checkpoint and raw training MIDI are downloaded separately and ignored
by Git. The download command verifies both source hashes; it downloads approximately
66 MB of checkpoint data and the 56 MB MIDI-only MAESTRO archive. It does not download
the 101 GB audio archive.

To reproduce extraction, training and evaluation from source data:

```powershell
.\.venv\Scripts\python.exe -m extension.run_demo --rebuild --render-audio
.\.venv\Scripts\python.exe -m unittest discover -s extension/tests -v
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

The five 30-second trials at seeds 229-233 measured **5.53 +/- 0.24 notes/sec** in sparse
mode and **12.41 +/- 0.05 notes/sec** in dense mode. Dense exceeded sparse in all five
pairs. These are extension results, not claims of reproducing the original paper's
metrics or measuring musical quality.

See [extension setup and methodology](extension/README.md), the
[experiment report](extension/report.md), and the [classroom demo outline](extension/presentation.md).

## Baseline replication

The baseline contains MATLAB Naive-Bayes-like and vanilla-network implementations,
a manually implemented NumPy LSTM, and a PyTorch encoder-decoder. Some original data
and model files are intentionally excluded from Git. The empty root `requirements.txt`
is not a complete environment specification for running all four methods; use the
extension requirements above for this feature.

Existing baseline runners use working-directory-specific paths. Do not run
`original_replication/lstm/LSTM.py` merely to launch the extension: it starts baseline
training. Existing baseline evidence remains in `results/evaluation/`.

## Sources

- [Original project paper and its public code-release link](https://cs229.stanford.edu/proj2018/report/18.pdf).
- [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro), used for the extension
  as substitute training data under CC BY-NC-SA 4.0. Attribution and selected
  performances are recorded in `extension/datasets/manifest.json`.

Develop extensions on a separate feature branch. Keep `main` as the recoverable
replication baseline; merge only after teammate review.
