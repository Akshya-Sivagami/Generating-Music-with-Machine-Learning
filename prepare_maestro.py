import os
import shutil
import glob

root = r"maestro_temp\maestro-v3.0.0"
data_dir = "data"

os.makedirs(data_dir, exist_ok=True)

# Remove the previously selected MIDI subset
for f in glob.glob(os.path.join(data_dir, "*.mid")):
    os.remove(f)

# Find every MIDI file in MAESTRO
midi_files = sorted(
    glob.glob(os.path.join(root, "**", "*.mid"), recursive=True)
)

# Use 100 MIDI performances
selected = midi_files[:100]

for i, source in enumerate(selected, 1):
    destination = os.path.join(data_dir, f"maestro_{i:03d}.mid")
    shutil.copy2(source, destination)

print(f"Found {len(midi_files)} MIDI files in MAESTRO.")
print(f"Copied {len(selected)} MIDI files to data\\")
