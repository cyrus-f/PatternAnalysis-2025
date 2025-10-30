import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
data = pd.read_csv("validation_data.csv")

# Extract columns
epochs = data["Epoch"]
train_loss = data["Training Loss"]
val_loss = data["Validation Loss"]
val_acc = data["Validation Accuracy"]

# === 1️⃣ Graph: Training and Validation Loss ===
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Training Loss", marker="o", linestyle="-")
plt.plot(epochs, val_loss, label="Validation Loss", marker="s", linestyle="--")
plt.title("Training vs Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_graph.png", dpi=300)
plt.show()

# === 2️⃣ Graph: Validation Accuracy ===
plt.figure(figsize=(10, 5))
plt.plot(epochs, val_acc, label="Validation Accuracy", color="orange", marker="o")
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_graph.png", dpi=300)
plt.show()