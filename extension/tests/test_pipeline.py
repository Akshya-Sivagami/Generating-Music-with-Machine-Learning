"""Timing, leakage, baseline-interface equivalence, and reproducibility checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np
import torch

from extension.common import ROOT, REPO, Note, group_onsets, read_midi, write_midi, midi_metrics

TEST_TMP = REPO / "tmp" / "tests"
TEST_TMP.mkdir(parents=True, exist_ok=True)


class MidiTests(unittest.TestCase):
    def test_tempo_map_and_velocity_zero_in_another_track(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            path = Path(folder) / "tempo.mid"
            midi = mido.MidiFile(type=1, ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage("set_tempo", tempo=500000),
                mido.MetaMessage("set_tempo", tempo=1000000, time=480),
                mido.MetaMessage("end_of_track", time=480)]))
            midi.tracks.append(mido.MidiTrack([
                mido.Message("note_on", note=60, velocity=80),
                mido.Message("note_on", note=60, velocity=0, time=960)]))
            midi.save(path)
            notes, length = read_midi(path)
            self.assertAlmostEqual(length, 1.5)
            self.assertAlmostEqual(notes[0].end, 1.5)
            self.assertEqual(midi_metrics(path)["note_count"], 1)

    def test_equal_duration_and_simultaneous_notes(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            path = Path(folder) / "chord.mid"
            notes = [Note(0, 0.5, 60), Note(0, 0.5, 64), Note(1, 1.5, 67)]
            write_midi(path, notes, 4)
            metrics = midi_metrics(path)
            self.assertEqual(metrics["note_count"], 3)
            self.assertEqual(metrics["notes_per_second"], 0.75)
            self.assertEqual(metrics["duration_seconds"], 4)
            self.assertEqual(metrics["mean_inter_onset_seconds"], 1)

    def test_grouping_does_not_chain_performed_notes(self):
        groups = group_onsets([Note(0, 1, 60), Note(0.02, 1, 64), Note(0.04, 1, 67)])
        self.assertEqual([len(g) for g in groups], [2, 1])

    def test_unclosed_note_and_type_two_fail_explicitly(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            path = Path(folder) / "invalid.mid"
            midi = mido.MidiFile()
            midi.tracks.append(mido.MidiTrack([mido.Message("note_on", note=60, velocity=80)]))
            midi.save(path)
            with self.assertRaisesRegex(ValueError, "Unclosed"):
                read_midi(path)
            midi.type = 2
            midi.save(path)
            with self.assertRaisesRegex(ValueError, "asynchronous"):
                read_midi(path)


@unittest.skipUnless((ROOT / "datasets/density.npz").exists(), "Rebuild the dataset to run split checks")
class DatasetTests(unittest.TestCase):
    def test_thresholds_fit_train_only_and_piece_splits_are_disjoint(self):
        with np.load(ROOT / "datasets/density.npz", allow_pickle=False) as data:
            stats = json.loads((ROOT / "datasets/statistics.json").read_text())
            train = data["split"] == "train"
            lower, upper = np.quantile(data["density"][train], [0.4, 0.6])
            self.assertEqual(lower, stats["sparse_max_notes_per_second"])
            self.assertEqual(upper, stats["dense_min_notes_per_second"])
            for a, b in [("train", "test"), ("train", "validation"), ("test", "validation")]:
                self.assertFalse(set(data["piece"][data["split"] == a]) & set(data["piece"][data["split"] == b]))
            expected = np.where(data["density"] <= lower, 0, np.where(data["density"] >= upper, 1, -1))
            np.testing.assert_array_equal(data["y"], expected)


@unittest.skipUnless((ROOT / "assets/baseline_checkpoint.pth").exists() and
                     (ROOT / "models/density_controller.joblib").exists(), "Download assets and train first")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from extension.encoder_adapter import load_baseline
        torch.set_num_threads(2)
        cls.loaded = load_baseline()

    def test_imported_encoder_decoder_matches_original_forward(self):
        from extension.encoder_adapter import seed_sequence
        from extension.generate_density_controlled import DEFAULT_SEED_MIDI
        source = REPO / "original_replication/encoder_decoder/encoderdecoder_model.py"
        spec = importlib.util.spec_from_file_location("original_for_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encoder, decoder, _ = self.loaded
        model = module.seq2seq(encoder, decoder, torch.device("cpu")).eval()
        src = seed_sequence(DEFAULT_SEED_MIDI)
        with torch.inference_mode():
            original = model(src, src, 0)
            hidden, cell = encoder(src.view(102, 1, 219))
            incoming = src[0]
            rows = []
            for _ in range(102):
                output, hidden, cell = decoder(incoming.view(1, 1, 219), hidden, cell)
                rows.append(output.reshape(-1))
                incoming = output
            torch.testing.assert_close(torch.stack(rows), original)

    def test_same_seed_byte_reproducibility_and_density_control(self):
        from extension.generate_density_controlled import generate
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            a, b, c = [Path(folder) / name for name in ["a.mid", "b.mid", "c.mid"]]
            sparse = generate("sparse", a, 5, 229, loaded=self.loaded)
            generate("sparse", b, 5, 229, loaded=self.loaded)
            dense = generate("dense", c, 5, 229, loaded=self.loaded)
            self.assertEqual(hashlib.sha256(a.read_bytes()).digest(), hashlib.sha256(b.read_bytes()).digest())
            self.assertGreater(dense["metrics"]["notes_per_second"], sparse["metrics"]["notes_per_second"])


if __name__ == "__main__":
    unittest.main()
