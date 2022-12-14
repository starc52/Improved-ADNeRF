import torch
import torch.nn as nn
from NeRFs.HeadNeRF.run_nerf_helpers import AudioNet, AudioAttNet
from LandmarkModels import LandmarkAutoencoder, LandmarkEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioConditionModel(nn.Module):
    def __init__(self, audnet_state=None,
                 audattnnet_state=None,
                 landmarkenc_state=None,
                 audnet_trainable=False,
                 landmarkenc_trainable=False,
                 eps=1e-5):
        super(AudioConditionModel, self).__init__()
        self.audionet = AudioNet(dim_aud=64)
        self.audioattnnet = AudioAttNet()
        self.landmark_encoder = LandmarkEncoder()
        self.cos_dist = nn.CosineSimilarity(dim=1)
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
        pos_audio_features = audio_features['pos'].to(device)
        pos_landmarks = landmarks['pos'].to(device)
        pos_audio_embs = self.audioattnnet(self.audionet(torch.squeeze(pos_audio_features, dim=0)))
        pos_audio_embs = torch.unsqueeze(pos_audio_embs, dim=0)
        pos_eye_embs, pos_mouth_embs = self.landmark_encoder(pos_landmarks)

        neg_audio_features = audio_features['neg'].to(device)
        neg_landmarks = landmarks['neg'].to(device)
        neg_audio_embs = self.audioattnnet(self.audionet(torch.squeeze(neg_audio_features, dim=0)))
        neg_audio_embs = torch.unsqueeze(neg_audio_embs, dim=0)
        neg_eye_embs, neg_mouth_embs = self.landmark_encoder(neg_landmarks)
        #print("pos_eye_embs", pos_eye_embs)
        #print("pos_mouth_embs", pos_mouth_embs)
        #print("pos_audio_embs", pos_audio_embs)
        #print("neg_eye_embs", neg_eye_embs)
        #print("neg_mouth_embs", neg_mouth_embs)
        #print("neg_audio_embs", neg_audio_embs)

        #print("pos cos dist", self.cos_dist(pos_audio_embs, pos_mouth_embs))
        #print("neg cos dist", self.cos_dist(neg_audio_embs, neg_mouth_embs))
        pos_cos = (self.cos_dist(pos_audio_embs, pos_mouth_embs)+1)/2
        neg_cos = (self.cos_dist(neg_audio_embs, neg_mouth_embs)+1)/2
        #print("pos log", -torch.sum(torch.nan_to_num(torch.log(pos_cos + self.eps))))
        #print("neg log", -torch.sum(torch.log(1 - neg_cos + self.eps)))
        contrastive_loss = -torch.sum(torch.nan_to_num(torch.log(pos_cos + self.eps))) - \
                          torch.sum(torch.log(1 - neg_cos + self.eps))
        return contrastive_loss
