import torch
import torch.nn as nn
from NeRFs.HeadNeRF.run_nerf_helpers import AudioNet, AudioAttNet
from LandmarkModels import LandmarkAutoencoder, LandmarkEncoder


class AudioConditioned(nn.Module):
    def __init__(self, audnet_state=None,
                 audattnnet_state=None,
                 landmarkenc_state=None,
                 audnet_trainable=False,
                 landmarkenc_trainable=False,
                 eps=1e-5):
        super(AudioConditioned, self).__init__()
        self.audionet = AudioNet()
        self.audioattnnet = AudioAttNet()
        self.landmark_encoder = LandmarkEncoder()
        self.cos_dist = nn.CosineSimilarity()
        self.eps = eps

        if audnet_state is not None:
            self.audionet.load_state_dict(audnet_state)

        if audattnnet_state is not None:
            self.audioattnnet.load_state_dict(audattnnet_state)

        if landmarkenc_state is not None:
            self.landmark_encoder.load_state_dict(landmarkenc_state)

        if not audnet_trainable:
            for param in self.audionet.parameters():
                param.requires_grad = False
            for param in self.audioattnnet.parameters():
                param.requires_grad = False
        else:
            for param in self.audionet.parameters():
                param.requires_grad = True
            for param in self.audioattnnet.parameters():
                param.requires_grad = True

        if not landmarkenc_trainable:
            for param in self.landmark_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.landmark_encoder.parameters():
                param.requires_grad = True

    def forward(self, audio_features, landmarks):
        pos_audio_features = audio_features['pos']
        pos_landmarks = landmarks['pos']
        pos_audio_embs = self.audioattnnet(self.audionet(pos_audio_features))
        pos_eye_embs, pos_mouth_embs = self.landmark_encoder(pos_landmarks)

        neg_audio_features = audio_features['neg']
        neg_landmarks = landmarks['neg']
        neg_audio_embs = self.audioattnnet(self.audionet(neg_audio_features))
        neg_eye_embs, neg_mouth_embs = self.landmark_encoder(neg_landmarks)

        contrastiveLoss = -torch.sum(torch.log(self.cos_dist(pos_audio_embs, pos_mouth_embs)+self.eps)) - \
                          torch.sum(torch.log(1 - self.cos_dist(neg_audio_embs, neg_mouth_embs)+self.eps))
        return contrastiveLoss