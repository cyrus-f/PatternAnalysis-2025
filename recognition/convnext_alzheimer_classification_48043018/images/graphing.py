import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV data ===
data_file = "validation_data.csv"
df = pd.read_csv(data_file)

# === Plot 1: Validation Loss vs Epoch ===
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['Validation Loss'], marker='o', linestyle='-', label='Validation Loss')
plt.title('Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('validation_loss_plot.png')
plt.show()

# === Plot 2: Validation Accuracy vs Epoch ===
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['Validation Accuracy'], color='orange', marker='o', linestyle='-', label='Validation Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('validation_accuracy_plot.png')
plt.show()