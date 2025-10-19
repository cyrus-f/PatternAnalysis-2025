from dataset import ADNIDataset
from modules import ConvNeXt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
"""
containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""
# CONSTANTS FOR ENVIRONMENT
LOCAL = 0
COLAB = 1
RANGPUR = 2

MACHINE = RANGPUR # change this depending on where you run the code
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
num_epochs = 30
learning_rate = 0.001
batch_size = 512
num_workers = 2 # number of subprocesses to use for data loading

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets and DataLoaders

# switch paths based on environment
train_dir = ['./ADNI/AD_NC/train',
              '/content/drive/MyDrive/ADNI/AD_NC/train',
                '/home/groups/comp3710/ADNI/AD_NC/train'][MACHINE]
test_dir = ['./ADNI/AD_NC/test', 
            '/content/drive/MyDrive/ADNI/AD_NC/test',
              '/home/groups/comp3710/ADNI/AD_NC/test'][MACHINE]

# Datasets
train_dataset = ADNIDataset(root_dir=train_dir, transform=train_transform)
test_dataset = ADNIDataset(root_dir=test_dir, transform=test_transform)

if __name__ == "__main__":
    # split train dataset into train and validation
    generator = torch.Generator().manual_seed(42) # for reproducibility so the split is always the same
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2], generator=generator) # do an 80-20 split for training and validation

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    
    # model, loss function, optimizer
    model = ConvNeXt().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch , content in enumerate(train_loader):
            images, labels = content[0].to(device), content[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * batch_size + len(images)
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{current}/{len(train_loader.dataset)}], Loss: {loss:.4f}")

        # validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                # accumulate validation statistics on-the-fly
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # If this is the last batch, compute and print average loss and accuracy
                if val_total == len(val_loader.dataset):
                    avg_val_loss = val_loss / val_total
                    val_accuracy = 100.0 * val_correct / val_total
                    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        torch.save(model.state_dict(), "model.pth")