import torch
import torch.nn as nn
import torch.nn.functional as F
from NeRFs.HeadNeRF.run_nerf_helpers import AudioNet, AudioAttNet
from LandmarkModels import LandmarkAutoencoder, LandmarkEncoder

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class AudioConditionModel(nn.Module):
    def __init__(self, audnet_state=None,
                 landmarkenc_state=None,
                 audnet_trainable=False,
                 landmarkenc_trainable=False,
                 eps=1e-5):
        super(AudioConditionModel, self).__init__()
        self.audionet = AudioNet(dim_aud=64)
        self.landmark_encoder = LandmarkEncoder()
        self.cos_dist = nn.CosineSimilarity(dim=1)
        self.eps = eps

        if audnet_state is not None:
            self.audionet.load_state_dict(audnet_state)

        if landmarkenc_state is not None:
            self.landmark_encoder.load_state_dict(landmarkenc_state)

        if not audnet_trainable:
            for param in self.audionet.parameters():
                param.requires_grad = False
        else:
            for param in self.audionet.parameters():
                param.requires_grad = True

        if not landmarkenc_trainable:
            for param in self.landmark_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.landmark_encoder.parameters():
                param.requires_grad = True

    @staticmethod
    def transpose(x):
        return x.transpose(-2, -1)

    @staticmethod
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError(
                    "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = self.normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ self.transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ self.transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ self.transpose(positive_key)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)

    def forward(self, query, positive_key, negative_key):
        positive_key_emb = positive_key.to(device)
        landmarks = query.to(device)
        positive_key_emb = self.audionet(positive_key_emb)
        query_eye_embs, query_mouth_embs = self.landmark_encoder(landmarks)

        negative_key_emb = negative_key.to(device)
        negative_key_emb = self.audionet(negative_key_emb)
        negative_key_emb = torch.unsqueeze(negative_key_emb, dim=1)
        
        contrastive_loss = self.info_nce(query_mouth_embs, positive_key_emb, negative_key_emb, negative_mode='paired')
        # #print("pos_eye_embs", pos_eye_embs)
        # #print("pos_mouth_embs", pos_mouth_embs)
        # #print("pos_audio_embs", pos_audio_embs)
        # #print("neg_eye_embs", neg_eye_embs)
        # #print("neg_mouth_embs", neg_mouth_embs)
        # #print("neg_audio_embs", neg_audio_embs)
        #
        # #print("pos cos dist", self.cos_dist(pos_audio_embs, pos_mouth_embs))
        # #print("neg cos dist", self.cos_dist(neg_audio_embs, neg_mouth_embs))
        # pos_cos = (self.cos_dist(pos_audio_embs, pos_mouth_embs)+1)/2
        # neg_cos = (self.cos_dist(neg_audio_embs, neg_mouth_embs)+1)/2
        # #print("pos log", -torch.sum(torch.nan_to_num(torch.log(pos_cos + self.eps))))
        # #print("neg log", -torch.sum(torch.log(1 - neg_cos + self.eps)))
        # contrastive_loss = -torch.sum(torch.nan_to_num(torch.log(pos_cos + self.eps))) - \
        #                   torch.sum(torch.log(1 - neg_cos + self.eps))
        return contrastive_loss
