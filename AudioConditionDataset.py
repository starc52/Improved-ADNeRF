import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random


class AudioConditionDataset(Dataset):
    """Audio Conditioning Dataset."""

    def __init__(self, csv_file, image_dim=224):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dim (int): size of images.
        """
        self.image_dim = image_dim
        self.landmarks_frame = pd.read_csv(csv_file, index_col=0)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        exact_row = self.landmarks_frame.iloc[idx]
        landmarks_a = torch.from_numpy(np.array(eval(exact_row['landmarks'])).astype(np.float32)) / self.image_dim

        frame_id = int(exact_row['image'].split('.')[0]) - 1
        audio_feat_path = '/'.join(exact_row['path'].split('/')[:-1]) + '.npy'
        audio_feat = torch.from_numpy(np.load(audio_feat_path).astype(np.float32))

        auds_pos = audio_feat[frame_id]

        sample_indice = list(range(audio_feat.shape[0]))
        removal_indice = range(max(0, frame_id - 7), min(frame_id + 7, audio_feat.shape[0] - 1))
        for index in removal_indice:
            sample_indice.remove(index)
        neg_index = random.choice(sample_indice)

        auds_neg = audio_feat[neg_index]

        query = landmarks_a
        positive_key = auds_pos
        negative_key = auds_neg
        # landmarks = {'pos': landmarks_a, 'neg': landmarks_a}
        # audio_features = {'pos': auds_pos, 'neg': auds_neg}
        return query, positive_key, negative_key
