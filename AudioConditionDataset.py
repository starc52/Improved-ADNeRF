import glob
import os
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
        removal_range = random.randint(45, 100)
        removal_indice = range(max(0, frame_id - removal_range), min(frame_id + removal_range, audio_feat.shape[0] - 1))
        for index in removal_indice:
            sample_indice.remove(index)
        if len(sample_indice) != 0:
            neg_index = random.choice(sample_indice)
        else:
            neg_index = -1
        auds_neg = audio_feat[neg_index]

        query = landmarks_a
        positive_key = auds_pos
        negative_key = auds_neg
        # landmarks = {'pos': landmarks_a, 'neg': landmarks_a}
        # audio_features = {'pos': auds_pos, 'neg': auds_neg}
        return query, positive_key, negative_key


class AudioConditionIdentityConstantDataset(Dataset):
    """Audio Conditioning Identity Constant Dataset."""

    def __init__(self, lms_path, aud_path, image_dim=450, dataset=0, partition=7272):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dim (int): size of images.
        """
        self.image_dim = image_dim
        self.landmarks_list = glob.glob(os.path.join(lms_path, '*.lms'))
        self.num_frames = len(self.landmarks_list)-1
        self.partition = partition
        self.dataset = dataset
        if self.dataset == 0:
            self.landmarks = np.array([np.loadtxt(os.path.join(lms_path, str(i)+'.lms')) for i in range(0, self.partition)]).astype(np.float32)
        else:
            self.landmarks = np.array([np.loadtxt(os.path.join(lms_path, str(i)+'.lms')) for i in range(self.partition, self.num_frames)]).astype(np.float32)
        self.aud_feat = np.load(aud_path).astype(np.float32)

    def __len__(self):
        return self.landmarks.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        landmarks_a = torch.from_numpy(self.landmarks[idx])/self.image_dim
        if self.dataset == 1:
            aud_idx = idx + self.partition
        else:
            aud_idx = idx
        # frame_id = int(exact_row['image'].split('.')[0]) - 1
        # audio_feat_path = '/'.join(exact_row['path'].split('/')[:-1]) + '.npy'
        # audio_feat = torch.from_numpy(np.load(audio_feat_path).astype(np.float32))

        auds_pos = self.aud_feat[aud_idx]
        if self.dataset == 0:
            sample_indice = list(range(0, self.partition))
        else:
            sample_indice = list(range(self.partition, self.num_frames))
        removal_range = random.randint(100, 500)
        removal_indice = range(max(0, aud_idx - removal_range), min(aud_idx + removal_range, self.num_frames))
        for index in removal_indice:
            sample_indice.remove(index)
        if len(sample_indice) != 0:
            neg_index = random.choice(sample_indice)
        else:
            neg_index = -1
        auds_neg = self.aud_feat[neg_index]

        query = landmarks_a
        positive_key = auds_pos
        negative_key = auds_neg
        # landmarks = {'pos': landmarks_a, 'neg': landmarks_a}
        # audio_features = {'pos': auds_pos, 'neg': auds_neg}
        return query, positive_key, negative_key