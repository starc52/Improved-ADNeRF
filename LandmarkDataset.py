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
        self.landmarks_frame = pd.read_csv(csv_file, index_col=0, na_filter=True, na_values='[]')
        self.landmarks_frame.dropna()
        self.landmarks_frame.astype(str)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        landmarks_a = np.array(eval(self.landmarks_frame.iloc[idx]['landmarks'])[0])/self.image_dim
        landmarks_a = landmarks_a.astype(np.float32)
        id_a = self.landmarks_frame.iloc[idx]['idx']
        video_a = self.landmarks_frame.iloc[idx]['video']
        image_a = self.landmarks_frame.iloc[idx]['image']
        spec_idx_a = self.landmarks_frame[self.landmarks_frame['idx'] == id_a]
        spec_video_a = spec_idx_a[spec_idx_a['video'] == video_a]
        spec_image_a = spec_video_a[spec_video_a['image']!=image_a]
        landmarks_b = eval(spec_image_a.iloc[0]['landmarks'])[0]
        landmarks_b = np.array(landmarks_b)/self.image_dim
        landmarks_b = landmarks_b.astype(np.float32)
        return torch.from_numpy(landmarks_a), torch.from_numpy(landmarks_b)
