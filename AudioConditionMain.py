import torch
from torch.utils.data import Dataset
from AudioConditionDataset import AudioConditionDataset
from AudioConditionModel import AudioConditionModel
import copy
import time
import wandb
from tqdm import tqdm

# wandb.init(project="Audio-Conditioning")

batch_size = 256
num_epochs = 4
switch_factor = 0.8
weight_decay = 1e-5
embedding_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.config = {"batch_size": batch_size,
#                 "epochs": num_epochs,
#                 "switch_factor": switch_factor,
#                 "weight_decay": weight_decay,
#                 }

train_audcond_dataset = AudioConditionDataset(csv_file='/scratch/tan/train_landmarks.csv')
val_audcond_dataset = AudioConditionDataset(csv_file='/scratch/tan/val_landmarks.csv')

dataset_sizes = {'train': len(train_audcond_dataset), 'val': len(val_audcond_dataset)}

train_dataloader = torch.utils.data.DataLoader(train_audcond_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=15)
val_dataloader = torch.utils.data.DataLoader(val_audcond_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=15)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

model = AudioConditionModel(audnet_trainable=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
since = time.time()
# wandb.watch(model)
best_loss = 1e10
for epoch in tqdm(range(num_epochs)):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for audio_features, landmarks in tqdm(dataloaders[phase]):
            audio_features = audio_features.to(device)
            landmarks = landmarks.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                contrastive_loss = model(audio_features, landmarks)
                # backward + optimize only if in training phase
                # wandb.log({"contrastive_loss": contrastive_loss})

                if phase == 'train':
                    contrastive_loss.backward()
                    optimizer.step()

            # statistics
            running_loss += contrastive_loss * landmarks['pos'].size(0)
        epoch_loss = running_loss / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f}')
        # wandb.log({phase + "_loss": epoch_loss})

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_audnet = copy.deepcopy(model.audionet.state_dict())
            best_audattnnet = copy.deepcopy(model.audioattnnet.state_dict())
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    torch.save(best_audnet, 'best_audnet.pt')
    torch.save(best_audattnnet, 'best_audattnnet.pt')
    torch.save(best_model_wts, 'best_audcond.pt')
    # load best model weights
    model.load_state_dict(best_model_wts)
