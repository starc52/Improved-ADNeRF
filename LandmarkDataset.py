import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, image_dim=224):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dim (int): size of images.
        """
        self.image_dim = image_dim
        self.landmarks_frame = pd.read_csv(csv_file, na_filter=True, na_values='[]')
        self.landmarks_frame.dropna()

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        landmarks_a = np.array(self.landmarks_frame.iloc[idx]['landmarks'])/self.image_dim
        video_a = self.landmarks_frame.iloc[idx]['video']
        image_a = self.landmarks_frame.iloc[idx]['image']
        landmarks_b = np.array(self.landmarks_frame[self.landmarks_frame['video'] == video_a & self.landmarks_frame['image'] != image_a].iloc[0]['landmarks'])/self.image_dim
        return torch.from_numpy(landmarks_a), torch.from_numpy(landmarks_b)
