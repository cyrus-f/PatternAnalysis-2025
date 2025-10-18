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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        pass