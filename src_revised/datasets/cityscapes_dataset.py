import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_pairs = self._get_frame_pairs()

    def _get_frame_pairs(self):
        """
        Retrieve all the consecutive frame pairs in the dataset.

        Returns:
            list: List of tuples containing paths to consecutive frame pairs.
        """
        frame_pairs = []
        for city_folder in sorted(os.listdir(self.root_dir)):
            city_path = os.path.join(self.root_dir, city_folder)
            if os.path.isdir(city_path):
                frames = sorted(os.listdir(city_path))
                for i in range(len(frames) - 1):
                    frame_pairs.append((os.path.join(city_path, frames[i]),
                                        os.path.join(city_path, frames[i+1])))
        return frame_pairs

    def __len__(self):
        """Returns the total number of frame pairs in the dataset."""
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the frame pair.

        Returns:
            tuple: (frame1, frame2) where frame1 and frame2 are consecutive frames.
        """
        frame1_path, frame2_path = self.frame_pairs[idx]

        # Load the images
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        # Convert from BGR to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL images for PyTorch transforms
        frame1 = Image.fromarray(frame1)
        frame2 = Image.fromarray(frame2)

        # Apply any preprocessing or augmentation
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)

        return frame1, frame2
