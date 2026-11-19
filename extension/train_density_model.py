"""Fit a diagnostic density classifier and an actually used conditional controller."""
import argparse
import json
import warnings

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from extension.common import ROOT, SEED, controller_features, save_json


def classifier_metrics(model, X, y):
    pred = model.predict(X)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix_sparse_dense": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "samples": len(y), "class_counts": np.bincount(y, minlength=2).tolist()}


def train(seed=SEED):
    with np.load(ROOT / "datasets" / "density.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    subsets = {s: (data["split"] == s) & (data["y"] >= 0) for s in ["train", "validation", "test"]}
    for s, mask in subsets.items():
        if len(np.unique(data["y"][mask])) != 2:
            raise ValueError(f"Both density classes are required in {s}; add more pieces")
    # Never split neighboring windows randomly. Both labels/normalizers are train-only.
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "small_mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8),
                       solver="lbfgs", alpha=1.0, max_iter=2000, random_state=seed)),
    }
    classifier_results = {}
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(data["X"][subsets["train"]], data["y"][subsets["train"]])
        classifier_results[name] = {s: classifier_metrics(model, data["X"][m], data["y"][m])
                                    for s, m in subsets.items()}
        classifier_results[name]["convergence_warnings"] = [str(w.message) for w in caught]
    # Choose by validation balanced accuracy, then report untouched test set.
    selected = max(["logistic_regression", "small_mlp"],
                   key=lambda n: classifier_results[n]["validation"]["balanced_accuracy"])
    indices = data["event_window"]
    event_labels = data["y"][indices]
    keep = event_labels >= 0
    contexts = data["event_context"][keep]
    labels = event_labels[keep]
    X = np.array([controller_features(c, label) for c, label in zip(contexts, labels)])
    targets = data["event_target"][keep]
    splits = data["split"][indices][keep]
    train_mask = splits == "train"
    # A compact learned multi-output controller. Logs keep all output quantities positive.
    controller = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    controller.fit(X[train_mask], np.log(targets[train_mask]))
    dummy = DummyRegressor(strategy="mean").fit(X[train_mask], np.log(targets[train_mask]))
    bounds = np.quantile(targets[train_mask], [0.01, 0.99], axis=0)
    controller_results = {}
    for s in ["train", "validation", "test"]:
        mask = splits == s
        actual = targets[mask]
        predicted = np.clip(np.exp(controller.predict(X[mask])), bounds[0], bounds[1])
        base = np.exp(dummy.predict(X[mask]))
        controller_results[s] = {
            "events": int(mask.sum()),
            "mae_gap_duration_chord_size": mean_absolute_error(actual, predicted, multioutput="raw_values").tolist(),
            "unconditional_mean_mae": mean_absolute_error(actual, base, multioutput="raw_values").tolist(),
        }
    models_path = ROOT / "models"
    models_path.mkdir(exist_ok=True)
    joblib.dump(models[selected], models_path / "density_classifier.joblib")
    joblib.dump({"model": controller, "bounds": bounds, "seed": seed,
                 "targets": ["onset_gap_seconds", "note_duration_seconds", "chord_size"]},
                models_path / "density_controller.joblib")
    report = {"seed": seed, "selected_classifier": selected,
              "selection_metric": "validation balanced accuracy", "classifiers": classifier_results,
              "controller": controller_results, "controller_architecture": "StandardScaler + multi-output Ridge on log targets; alpha=10",
              "controller_output_bounds_train_1st_99th_percentile": bounds.tolist(),
              "classifier_role": "Diagnostic; not the generative control. The conditional ridge controller drives decoding.",
              "feature_warning": "Classifier excludes note count/density and current inter-onset gaps; duration and chord size remain correlated with labels.",
              "packages": {"numpy": np.__version__, "sklearn": __import__("sklearn").__version__,
                           "torch": __import__("torch").__version__, "mido": __import__("mido").version_info}}
    # NamedTuple version_info needs a plain string for JSON.
    report["packages"]["mido"] = str(report["packages"]["mido"])
    save_json(ROOT / "outputs" / "model_evaluation.json", report)
    print(json.dumps({"selected_classifier": selected,
                     "test": classifier_results[selected]["test"],
                     "controller_test": controller_results["test"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    train(parser.parse_args().seed)


if __name__ == "__main__":
    main()
