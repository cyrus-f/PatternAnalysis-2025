# parser.py

import csv
import re

input_file = "training_log.txt"
output_file = "validation_data.csv"
with open(input_file, "r") as f:
    for _ in range(5):
        print(repr(f.readline()))

# Regex to extract validation loss and accuracy
pattern = re.compile(r"[Vv]alidation\s*Loss:\s*([0-9]*\.?[0-9]+)[,\s]+Accuracy:\s*([0-9]*\.?[0-9]+)")
validation_data = []

with open(input_file, "r") as f:
    lines = f.readlines()

# Iterate through all lines and extract matches
for line in lines:
    match = pattern.search(line)
    if match:
        val_loss = float(match.group(1))
        val_acc = float(match.group(2))
        validation_data.append((val_loss, val_acc))

# Write to CSV
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Epoch", "Validation Loss", "Validation Accuracy"])
    for epoch, (loss, acc) in enumerate(validation_data, start=1):
        writer.writerow([epoch, loss, acc])

print(f"✅ Parsed {len(validation_data)} validation entries and saved to '{output_file}'")

