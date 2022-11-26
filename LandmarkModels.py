import torch
import torch.nn as nn


class LandmarkEncoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__(self)
        self.linear1 = nn.Linear(in_features=68*2, out_features=256)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.02)

        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.02)

        self.linear3 = nn.Linear(in_features=256, out_features=128)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.02)

        self.linear4_eye = nn.Linear(in_features=128, out_features=embedding_size)
        self.linear4_mouth = nn.Linear(in_features=128, out_features=embedding_size)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, landmarks):
        landmarks_flat = landmarks.view(-1, 68*2)

        x = self.linear1(landmarks_flat)
        x = self.lrelu1(x)

        x = self.linear2(x)
        x = self.lrelu2(x)

        x = self.linear3(x)
        x = self.lrelu3(x)

        x_eye = torch.clone(x)
        x_eye = self.linear4_eye(x_eye)
        x_eye = self.lrelu4(x_eye)

        x_mouth = self.linear4_mouth(x)
        x_mouth = self.lrelu4(x_mouth)
        return x_eye, x_mouth


class LandmarkDecoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__(self)
        self.linear5 = nn.Linear(in_features=2*embedding_size, out_features=256)
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.02)

        self.linear6 = nn.Linear(in_features=256, out_features=256)
        self.lrelu6 = nn.LeakyReLU(negative_slope=0.02)

        self.linear7 = nn.Linear(in_features=256, out_features=256)
        self.lrelu7 = nn.LeakyReLU(negative_slope=0.02)

        self.linear8 = nn.Linear(in_features=256, out_features=68 * 2)
        self.lrelu8 = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, eye_emb, mouth_emb):

        x = torch.cat((eye_emb, mouth_emb), dim=1)

        x = self.linear5(x)
        x = self.lrelu5(x)

        x = self.linear6(x)
        x = self.lrelu6(x)

        x = self.linear7(x)
        x = self.lrelu7(x)

        x = self.linear8(x)
        x = self.lrelu8(x)

        pred = x.view(-1, 68, 2)

        return pred


class LandmarkAutoencoder(nn.Module):
    def __init__(self, switch_factor):
        super().__init__(self)
        self.switch_factor = switch_factor
        self.encoder = LandmarkEncoder(embedding_size=64)
        self.decoder = LandmarkDecoder(embedding_size=64)
        self.pairwisedist = nn.PairwiseDistance(eps=0)

    def forward(self, landmarks_a, landmarks_b):

        eye_emb_a, mouth_emb_a = self.encoder(landmarks_a)
        eye_emb_b, mouth_emb_b = self.encoder(landmarks_b)

        switch = torch.randint(low=0, high=2, size=(landmarks_a.size(0),))

        x_mouth_a_switch = torch.where(switch == 1, mouth_emb_b, mouth_emb_a)
        x_mouth_b_switch = torch.where(switch == 1, mouth_emb_a, mouth_emb_b)

        pred_a = self.decoder(eye_emb_a, x_mouth_a_switch)
        pred_b = self.decoder(eye_emb_b, x_mouth_b_switch)

        temp_land = torch.clone(landmarks_a)
        landmarks_a[:, 48:, :] = torch.where(switch == 1, landmarks_b[:, 48:, :], landmarks_a[:, 48:, :])
        landmarks_b[:, 48:, :] = torch.where(switch == 1, temp_land[:, 48:, :], landmarks_b[:, 48:, :])

        l2_loss_a = self.pairwisedist(landmarks_a, pred_a)
        l2_loss_b = self.pairwisedist(landmarks_b, pred_b)

        recon_loss_a = torch.mean(torch.sum(l2_loss_a, dim=1))
        recon_loss_b = torch.mean(torch.sum(l2_loss_b, dim=1))

        return recon_loss_a, recon_loss_b, recon_loss_a+recon_loss_b
