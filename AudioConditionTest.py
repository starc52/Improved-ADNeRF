import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from LandmarkModels import LandmarkAutoencoder, LandmarkEncoder
from AudioConditionDataset import AudioConditionDataset
from AudioConditionModel import AudioConditionModel
from NeRFs.HeadNeRF.run_nerf_helpers import AudioNet, AudioAttNet
import copy
import time
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay

wandb.init(project="Audio-Conditioning-Test")

batch_size = 1024
accumulation = 1
num_epochs = 20
weight_decay = 1e-5
embedding_size = 64
lrate_decay = 2500
lrate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.config = {"batch_size": batch_size,
                "accumulation": accumulation,
                "embedding_size": embedding_size,
                "epochs": num_epochs,
                "weight_decay": weight_decay,
                }

train_audcond_dataset = AudioConditionDataset(csv_file='/scratch/tan/train_landmarks1p.csv')
val_audcond_dataset = AudioConditionDataset(csv_file='/scratch/tan/val_landmarks1p.csv')

dataset_sizes = {'train': len(train_audcond_dataset), 'val': len(val_audcond_dataset)}

train_dataloader = torch.utils.data.DataLoader(train_audcond_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_audcond_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

landmark_autoencoder_state = torch.load('./best_autoencoder.pt')
autoencoder = LandmarkAutoencoder(switch_factor=0.8, embedding_size=64)
autoencoder.load_state_dict(landmark_autoencoder_state)
landmark_encoder_state = autoencoder.encoder.state_dict()

model = AudioConditionModel(landmarkenc_state=landmark_encoder_state, landmarkenc_trainable=False,
                            audnet_trainable=False).to(device)

model.load_state_dict(torch.load('./best_audcond.pt'))

landmark_encoder_state = model.landmark_encoder.state_dict()
landmark_encoder = LandmarkEncoder(embedding_size=64)
landmark_encoder.load_state_dict(landmark_encoder_state)

audnet_state = model.audionet.state_dict()
audnet = AudioNet(dim_aud=64)
audnet.load_state_dict(audnet_state)


class BinaryClassifier(nn.Module):
    def __init__(self, landmark_enc, audionet):
        self.landmark_encoder = landmark_enc
        self.audnet = audionet
        self.landmark_encoder.eval()
        self.audnet.eval()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=64 * 2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )
        self.bceloss = nn.BCELoss()

    def forward(self, query, positive, negative):
        positive_emb = positive.to(device)
        landmarks = query.to(device)
        positive_emb = self.audnet(positive_emb)
        positive_emb = torch.unsqueeze(positive_emb, dim=1)
        query_eye_embs, query_mouth_embs = self.landmark_encoder(landmarks)

        negative_emb = negative.to(device)
        negative_emb = self.audnet(negative_emb)
        negative_emb = torch.unsqueeze(negative_emb, dim=1)

        pos_pair = torch.cat((query_mouth_embs, positive_emb), dim=1)
        neg_pair = torch.cat((query_mouth_embs, negative_emb), dim=1)

        pos_x = self.mlp(pos_pair)
        pos_labels = torch.ones_like(pos_x)

        neg_x = self.mlp(neg_pair)
        neg_labels = torch.zeros_like(neg_x)

        pred = torch.squeeze(torch.vstack([pos_x, neg_x]))
        labels = torch.squeeze(torch.vstack([pos_labels, neg_labels]))

        loss = self.bceloss(pred, labels)
        return loss, pred, labels


bin_classifier = BinaryClassifier(landmark_enc=landmark_encoder, audionet=audnet)

optimizer = torch.optim.Adam(bin_classifier.parameters(), lr=lrate, weight_decay=weight_decay)
since = time.time()
wandb.watch(bin_classifier, log_freq=100)
best_loss = 1e10
global_step = 0
epoch_loss = {'train': 0.0, 'val': 0.0}
for epoch in tqdm(range(num_epochs)):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    for phase in ['train', 'val']:
        if phase == 'train':
            bin_classifier.train()  # Set model to training mode
            # zero the parameter gradients
            optimizer.zero_grad()
        else:
            bin_classifier.eval()  # Set model to evaluate mode

        running_loss = {'train': 0.0, 'val': 0.0}
        running_corrects = 0
        actual_loss = 0.0
        # Iterate over data.
        pred = []
        label = []
        for batch_id, (query, positive_key, negative_key) in enumerate(tqdm(dataloaders[phase])):
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                loss, preds, labels = bin_classifier(query, positive_key, negative_key)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss = loss / accumulation
                    actual_loss += loss
                    loss.backward()
                    if (batch_id + 1) % accumulation == 0 or batch_id == len(dataloaders[phase]) - 1:
                        wandb.log({"train_contrastive_loss": actual_loss})
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    pred.extend(preds.cpu().detach().numpy().tolist())
                    label.extend(labels.cpu().detach().numpy().tolist())
                    wandb.log({'val_contrastive_loss': loss})

            # statistics
            if phase == 'train':
                running_loss[phase] += actual_loss * query.size(0)
                actual_loss = 0.0
            else:
                running_loss[phase] += loss * query.size(0)

            if phase == 'train':
                decay_rate = 0.1
                decay_steps = lrate_decay
                new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                global_step += 1
        pred = np.array(pred)
        label = np.array(label)
        wandb.log({f"roc_score_{phase}": roc_auc_score(label, pred)})
        RocCurveDisplay.from_predictions(label, pred)
        plt.savefig(f"./roc_curve_{phase}.jpg")
        path_to_img = f"./roc_curve_{phase}.jpg"
        im = plt.imread(path_to_img)
        wandb.log({f"roc_curve_{phase}": [wandb.Image(im, caption=f"ROC curve on {phase} set for {epoch}")]})
        plt.clf()

        epoch_loss[phase] = running_loss[phase] / dataset_sizes['train']
        epoch_loss[phase] = running_loss[phase] / dataset_sizes['val']
        print(f'{phase} Loss: {epoch_loss[phase]:.4f}')
        wandb.log({phase + "_loss": epoch_loss[phase]})

        # deep copy the model
        if phase == 'val' and epoch_loss[phase] < best_loss:
            best_loss = epoch_loss[phase]
            best_model_wts = copy.deepcopy(bin_classifier.state_dict())
            torch.save(best_model_wts, 'best_bin_classifier.pt')

    print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    torch.save(best_model_wts, 'best_bin_classifier.pt')
    # load best model weights
    bin_classifier.load_state_dict(best_model_wts)
