import pandas as pd
import os
import shutil

root = r"maestro_temp\maestro-v3.0.0"
csv = os.path.join(root, "maestro-v3.0.0.csv")

df = pd.read_csv(csv)

x = df[
    df["canonical_composer"].str.contains("Chopin", case=False, na=False) &
    df["canonical_title"].str.contains("Etude", case=False, na=False)
].copy()

# Keep only individually named Etudes, not complete/grouped recordings
x = x[
    ~x["canonical_title"].str.contains(
        "12 Etudes|Complete Etudes|Nos\\. 1-5",
        case=False,
        regex=True,
        na=False
    )
].copy()

# Extract Opus and Etude number
x["opus"] = x["canonical_title"].str.extract(r"Op\.\s*(10|25)", expand=False)
x["number"] = x["canonical_title"].str.extract(r"No\.?\s*(\d+)", expand=False)

x = x.dropna(subset=["opus", "number"])

# One recording per Etude
x["key"] = x["opus"] + "-" + x["number"]
x = x.drop_duplicates("key")

os.makedirs("data", exist_ok=True)

print("\nSELECTED CHOPIN ETUDES:\n")

count = 0

for _, row in x.iterrows():
    source = os.path.join(root, row["midi_filename"])

    if not os.path.exists(source):
        print("MISSING:", source)
        continue

    filename = f"Chopin_Op{row['opus']}_No{int(row['number'])}.mid"
    destination = os.path.join("data", filename)

    shutil.copy2(source, destination)

    print(f"{filename} <- {row['canonical_title']}")
    count += 1

print(f"\nCopied {count} MIDI files into data\\")
