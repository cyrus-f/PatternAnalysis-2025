"""predict.py
Key components for performing inference using the trained ConvNeXt model for Alzheimer's classification.
This script includes loading the trained model, preparing the test dataset, performing inference,
and displaying the results including a confusion matrix.
"""
from dataset import ADNIDataset
from modules import ConvNeXt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# CONSTANTS FOR ENVIRONMENT
LOCAL = 0
COLAB = 1
RANGPUR = 2

MACHINE = LOCAL # change this depending on where you run the code

# Device configuration
if torch.cuda.is_available(): # for GPU users
    device = torch.device('cuda')
elif torch.backends.mps.is_available():  # for mac users with M series chips
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# Class labels
NC = 0
AD = 1

# Hyperparameters
batch_size = 32
num_workers = 2 if MACHINE == COLAB else (1 if MACHINE == RANGPUR else 4) # number of subprocesses to use for data loading

# Load the trained model
model = ConvNeXt()
model.load_state_dict(torch.load("model.pth", map_location=device)) # load trained model weights
model = model.to(device)
model.eval()

# Data transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalise the image
])

# Datasets and DataLoaders

# switch paths based on environment
test_dir = ['./ADNI/AD_NC/test', 
            '/content/ADNI/AD_NC/test',
              '/home/groups/comp3710/ADNI/AD_NC/test'][MACHINE]

# Datasets
print("Loading datasets...")
test_dataset = ADNIDataset(root_dir=test_dir, transform=test_transform)
print("Datasets loaded.")

if __name__ == "__main__":
    print("Creating data loaders...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("Data loaders created.")
    # Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # 2-class classification
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Print simple accuracy
    accuracy = (all_preds == all_labels).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), labels=[NC, AD])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NC', 'AD'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set")
    plt.savefig('images/confusion_matrix.png')
    print("Confusion matrix saved to images/confusion_matrix.png")
    plt.show()