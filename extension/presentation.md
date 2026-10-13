# Classroom presentation and live demo outline

## 1. Problem and prior work

Title: **Controllable Music Generation using Learned Note Density**.

The project replicates four Stanford CS229 approaches: Naive-Bayes-like generation,
vanilla neural network, LSTM and encoder-decoder. The completed replication is kept
intact. Our contribution adds a sparse/dense choice to one existing generator.
Describe sparse as fewer note-on events per second and dense as more. Avoid equating
density with musical complexity or quality.

## 2. What is learned?

Show `plots/training_distribution.png`. Explain four-second MIDI windows and
training-only thresholds: sparse <= 7.25, dense >= 11.00 notes/sec. Training uses
16 MAESTRO Chopin performances; validation and test each use six performances from
the official separate composition partitions. Neighboring windows stay together.

Explain the controller: mode + previous musical context -> learned onset gap,
duration and chord size. It is a small ridge model over log targets. The original
pretrained encoder-decoder supplies pitches. The event is chosen inside decoding
and feeds the next step. No finished MIDI is edited by random deletion/insertion.

## 3. Two evaluations

Diagnostic classifier: logistic regression selected by validation balanced accuracy;
85.49% test accuracy versus 60.39% majority accuracy. Explain that the diagnostic
classifier does not drive generation; the conditional ridge controller does.

Controller: held-out onset-gap MAE 0.07884 s versus 0.08392 s unconditional mean;
duration/chord-size errors also improve. These are real measured values and modest
improvements, not evidence of musical preference.

## 4. Live demonstration

Prepare beforehand from the root in the activated virtual environment:

```text
python -m extension.download_assets
python -m extension.run_demo --render-audio
```

During evaluation, generate each mode:

```text
python -m extension.run_demo --mode sparse --render-audio
python -m extension.run_demo --mode dense --render-audio
```

Play `extension/outputs/sparse_generated.wav`, then `dense_generated.wav`, using any
audio player. MIDI can be played with a piano synthesizer instead. WAV preview tones
are deliberately simple; demonstrate event density rather than instrument quality.
The seed-229 pair contains 163 versus 373 notes over the same 30 seconds.

Have the two generated files ready as an offline fallback; downloads and training
are not required during the classroom session after setup.

## 5. Show control across several seeds

Show `plots/density_comparison.png` and `plots/piano_roll.png`. Sparse measures
5.53 +/- 0.24 notes/sec; dense 12.41 +/- 0.05, across seeds 229-233 (sample SD).
Dense exceeds sparse in all five paired trials. The checkpoint, context, duration
and sampling settings match, with only the requested mode changing.

## 6. Limitations and questions

Discuss the small data subset, one generation seed context, correlated windows,
smooth regression timing and frozen pitch generator. Density control does not
guarantee beautiful music. Human listening and teammate review are still required.

Be ready to show `extension/generate_density_controlled.py` (controller use before
emission), `extract_density_dataset.py` (training-only thresholds),
`outputs/model_evaluation.json` (all model results) and
`outputs/density_comparison.csv` (measurements of saved MIDI).
