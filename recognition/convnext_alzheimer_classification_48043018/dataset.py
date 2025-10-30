""" dataset.py
Custom Dataset class for loading ADNI images and labels.
This module defines the ADNIDataset class which inherits from torch.utils.data.Dataset.
It handles loading images from specified directories, applying transformations, and providing
image-label pairs for training and evaluation.

key classes:
- ImageInfo: A simple class to hold image path and label information.
- ADNIDataset: Custom Dataset for loading ADNI images and labels.
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageInfo:
    """
    Class to hold information about each image and its label.
    Args:
        image_path (str): Path to the image file.
        label (int): Label associated with the image.
    Attrs:
        image_path (str): Path to the image file.
        label (int): Label associated with the image.
    """
    def __init__(self, image_path, label):
        self.image_path = image_path
        self.label = label
class ADNIDataset(Dataset):
    """
    Custom Dataset for loading ADNI images and labels.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_infos = []
        for label, subdir in enumerate(['NC', 'AD']):
            subdir_path = os.path.join(root_dir, subdir)
            for img_name in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_name)
                self.image_infos.append(ImageInfo(img_path, label))

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.image_infos)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, label) where image is a transformed image tensor and label is its corresponding label.
        """
        img_info = self.image_infos[idx]
        image = Image.open(img_info.image_path).convert('RGB')
        label = img_info.label

        if self.transform:
            image = self.transform(image)

        return image, label