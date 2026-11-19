"""Fetch verified public assets and extract a small, official-split Chopin subset."""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

from extension.common import ROOT, SEED, save_json

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_SHA = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"
CHECKPOINT_URL = ("https://www.dropbox.com/scl/fo/6e4f4ho701jbvfa3u67g1/"
                  "APgIc7_2LqPKO7B7OFpTlNI/Encoder%20Decoder/105modelstate.pth"
                  "?rlkey=7n4j96pb8p3j6kmhnb8qc1l7b&dl=1")
CHECKPOINT_SHA = "8cee9b78f7b6e86ccc3c891a0ba7a6eabf531f1d5eb3eeff702f029475e50c28"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == expected:
        return
    temporary = path.with_suffix(path.suffix + ".download")
    print(f"Downloading {path.name}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    if sha256(temporary) != expected:
        temporary.unlink()
        raise ValueError(f"Checksum mismatch: {path.name}")
    temporary.replace(path)


def prepare_assets(train=16, validation=6, test=6):
    assets = ROOT / "assets"
    download(CHECKPOINT_URL, assets / "baseline_checkpoint.pth", CHECKPOINT_SHA)
    archive = assets / "maestro-v3.0.0-midi.zip"
    download(MAESTRO_URL, archive, MAESTRO_SHA)
    selected = []
    with zipfile.ZipFile(archive) as z:
        rows = list(csv.DictReader(io.StringIO(z.read("maestro-v3.0.0/maestro-v3.0.0.csv").decode())))
        for split, limit in [("train", train), ("validation", validation), ("test", test)]:
            candidates = sorted((r for r in rows if "Chopin" in r["canonical_composer"]
                                 and r["split"] == split),
                                key=lambda r: (hashlib.sha256(f"{SEED}:{r['canonical_title']}".encode()).hexdigest(),
                                               float(r["duration"]), r["midi_filename"]))
            titles = set()
            for row in candidates:
                title = row["canonical_title"]
                if title in titles:
                    continue
                titles.add(title)
                index = len([r for r in selected if r["split"] == split])
                relative = f"{split}/{index:02d}.midi"
                output = assets / "midi" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(z.read("maestro-v3.0.0/" + row["midi_filename"]))
                selected.append({**row, "local_path": relative, "sha256": sha256(output)})
                if len(titles) >= limit:
                    break
            if len(titles) < limit:
                raise ValueError(f"Only {len(titles)} distinct Chopin titles in {split}")
    save_json(ROOT / "datasets" / "manifest.json", {
        "dataset": "MAESTRO v3.0.0, Chopin subset", "seed": SEED,
        "source": "https://magenta.tensorflow.org/datasets/maestro",
        "license": "CC BY-NC-SA 4.0", "archive_sha256": MAESTRO_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA, "checkpoint_source": CHECKPOINT_URL,
        "split_policy": "Official MAESTRO composition-disjoint splits; seeded title-hash order, one performance per exact title",
        "files": selected,
    })
    print(f"Verified checkpoint and {len(selected)} MIDI performances")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pieces", type=int, default=16)
    parser.add_argument("--validation-pieces", type=int, default=6)
    parser.add_argument("--test-pieces", type=int, default=6)
    args = parser.parse_args()
    if min(args.train_pieces, args.validation_pieces, args.test_pieces) < 1:
        parser.error("Each split must contain at least one piece")
    prepare_assets(args.train_pieces, args.validation_pieces, args.test_pieces)


if __name__ == "__main__":
    main()
