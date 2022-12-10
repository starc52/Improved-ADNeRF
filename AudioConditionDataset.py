import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random


class AudioConditionDataset(Dataset):
    """Audio Conditioning Dataset."""

    def __init__(self, csv_file, image_dim=224, smooth_win=8):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dim (int): size of images.
        """
        self.image_dim = image_dim
        self.smooth_win = smooth_win
        self.landmarks_frame = pd.read_csv(csv_file, index_col=0)
        self.landmarks_frame = self.landmarks_frame[self.landmarks_frame['idx'] == 'id00755'] # remove this comment after testing

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        exact_row = self.landmarks_frame.iloc[idx]
        landmarks_a = torch.from_numpy(np.array(exact_row['landmarks'])) / self.image_dim

        frame_id = int(exact_row['image'].split('.')[0]) - 1
        audio_feat_path = '/'.join(exact_row['path'].split('/')[:-1]) + '.npy'
        audio_feat = torch.from_numpy(np.load(audio_feat_path).astype(np.float32))

        # for smoothing around positive sample
        smo_half_win = int(self.smooth_win / 2)
        left_i = frame_id - smo_half_win
        right_i = frame_id + smo_half_win
        pad_left, pad_right = 0, 0
        if left_i < 0:
            pad_left = -left_i
            left_i = 0
        if right_i > audio_feat.shape[0]:
            pad_right = right_i - audio_feat.shape[0]
            right_i = audio_feat.shape[0]
        auds_pos = audio_feat[left_i:right_i]
        if pad_left > 0:
            auds_pos = torch.cat(
                (torch.zeros_like(auds_pos)[:pad_left], auds_pos), dim=0)
        if pad_right > 0:
            auds_pos = torch.cat(
                (auds_pos, torch.zeros_like(auds_pos)[:pad_right]), dim=0)

        sample_indice = list(range(audio_feat.shape[0]))
        removal_indice = range(max(0, frame_id - 8), min(frame_id + 8, audio_feat.shape[0] - 1))
        for index in removal_indice:
            sample_indice.remove(index)
        neg_index = random.choice(sample_indice)

        # for smoothing around negative sample
        smo_half_win = int(self.smooth_win / 2)
        left_i = neg_index - smo_half_win
        right_i = neg_index + smo_half_win
        pad_left, pad_right = 0, 0
        if left_i < 0:
            pad_left = -left_i
            left_i = 0
        if right_i > audio_feat.shape[0]:
            pad_right = right_i - audio_feat.shape[0]
            right_i = audio_feat.shape[0]
        auds_neg = audio_feat[left_i:right_i]
        if pad_left > 0:
            auds_neg = torch.cat(
                (torch.zeros_like(auds_neg)[:pad_left], auds_neg), dim=0)
        if pad_right > 0:
            auds_neg = torch.cat(
                (auds_neg, torch.zeros_like(auds_neg)[:pad_right]), dim=0)

        landmarks = {'pos': landmarks_a, 'neg': landmarks_a}
        audio_features = {'pos': auds_pos, 'neg': auds_neg}
        return audio_features, landmarks
