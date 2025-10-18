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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyperparameters
num_epochs = 25
learning_rate = 0.001
batch_size = 16

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

# absolute paths: uncomment if running online
#train_dir = '/PatternAnalysis-2025/recognition/convnext_alzheimer_classification_48043018/ADNI/AD_NC/train'
#test_dir = '/PatternAnalysis-2025/recognition/convnext_alzheimer_classification_48043018/ADNI/AD_NC/test'

# relative paths: uncomment if running locally
train_dir = './ADNI/AD_NC/train'
test_dir = './ADNI/AD_NC/test'


train_dataset = ADNIDataset(root_dir=train_dir, transform=train_transform)
test_dataset = ADNIDataset(root_dir=test_dir, transform=test_transform)

if __name__ == "__main__":
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    batch = next(iter(train_loader))
    images, labels = batch
    print(f'Image batch shape: {images.size()}')
    print(f'Label batch shape: {labels.size()}')
    # display first image in batch
    img = images[0].permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(f'Label: {labels[0].item()}')
    plt.show()