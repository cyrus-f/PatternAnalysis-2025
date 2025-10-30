import re
import csv

input_file = "training_log.txt"
output_file = "validation_data.csv"

# Regular expressions
val_pattern = re.compile(r"Validation Loss:\s*([\d.]+), Accuracy:\s*([\d.]+)%")
train_pattern = re.compile(r"Epoch \[(\d+)/\d+\], Step \[16768/17262\], Loss:\s*([\d.]+)")

epochs = []
train_losses = []
val_losses = []
val_accuracies = []

with open(input_file, "r") as f:
    lines = f.readlines()

current_epoch = 0
for line in lines:
    line = line.strip()

    # Match the final training loss for an epoch
    train_match = train_pattern.match(line)
    if train_match:
        current_epoch = int(train_match.group(1))
        train_loss = float(train_match.group(2))
        train_losses.append(train_loss)
        continue

    # Match the validation results for the same epoch
    val_match = val_pattern.match(line)
    if val_match:
        val_loss = float(val_match.group(1))
        val_acc = float(val_match.group(2))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        epochs.append(len(epochs) + 1)

# Ensure all lists are aligned
num_epochs = min(len(epochs), len(train_losses), len(val_losses), len(val_accuracies))
epochs = epochs[:num_epochs]
train_losses = train_losses[:num_epochs]
val_losses = val_losses[:num_epochs]
val_accuracies = val_accuracies[:num_epochs]

# === Write CSV ===
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Validation Accuracy"])
    for i in range(num_epochs):
        writer.writerow([epochs[i], train_losses[i], val_losses[i], val_accuracies[i]])

print(f"✅ Parsed {num_epochs} epochs and saved to '{output_file}'")