import torch
from torch.utils.data import Dataset
from LandmarkModels import LandmarkAutoencoder, LandmarkEncoder
from AudioConditionDataset import AudioConditionDataset
from AudioConditionModel import AudioConditionModel
import copy
import time
import wandb
from tqdm import tqdm

wandb.init(project="Audio-Conditioning")

batch_size = 32
accumulation = 1
num_epochs = 4
weight_decay = 1e-5
embedding_size = 64
lrate_decay = 250
lrate = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.config = {"batch_size": batch_size,
                "accumulation": accumulation,
                "embedding_size": embedding_size,
                "epochs": num_epochs,
                "weight_decay": weight_decay,
                }

train_audcond_dataset = AudioConditionDataset(csv_file='/scratch/tan/train_landmarks10p.csv')
val_audcond_dataset = AudioConditionDataset(csv_file='/scratch/tan/val_landmarks10p.csv')

dataset_sizes = {'train': len(train_audcond_dataset), 'val': len(val_audcond_dataset)}

train_dataloader = torch.utils.data.DataLoader(train_audcond_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1)
val_dataloader = torch.utils.data.DataLoader(val_audcond_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

landmark_autoencoder_state = torch.load('./best_autoencoder.pt')
autoencoder = LandmarkAutoencoder(switch_factor=0.8, embedding_size=64)
landmark_encoder_state = autoencoder.encoder.state_dict()

model = AudioConditionModel(landmarkenc_state=landmark_encoder_state, audnet_trainable=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)
since = time.time()
wandb.watch(model)
best_loss = 1e10
global_step = 0
epoch_loss = {'train': 0.0, 'val': 0.0}
for epoch in tqdm(range(num_epochs)):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            # zero the parameter gradients
            optimizer.zero_grad()
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = {'train': 0.0, 'val': 0.0}
        running_corrects = 0
        actual_loss=0.0
        # Iterate over data.
        for batch_id, (query, positive_key, negative_key) in enumerate(tqdm(dataloaders[phase])):
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                contrastive_loss = model(query, positive_key, negative_key)
                # backward + optimize only if in training phase
                if phase == 'train':
                    contrastive_loss = contrastive_loss/accumulation
                    actual_loss += contrastive_loss
                    contrastive_loss.backward()
                    if (batch_id+1)%accumulation == 0 or batch_id == len(dataloaders[phase])-1:
                        wandb.log({"train_contrastive_loss": actual_loss})
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    wandb.log({'val_contrastive_loss': contrastive_loss})

            # statistics
            if phase == 'train':
                running_loss[phase] += actual_loss * query.size(0)
                actual_loss = 0.0
            else:
                running_loss[phase] += contrastive_loss * query.size(0)

            if phase == 'train':
                decay_rate = 0.1
                decay_steps = lrate_decay * 1000
                new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                global_step += 1
        epoch_loss[phase] = running_loss[phase] / dataset_sizes['train']
        epoch_loss[phase] = running_loss[phase] / dataset_sizes['val']
        print(f'{phase} Loss: {epoch_loss[phase]:.4f}')
        wandb.log({phase + "_loss": epoch_loss[phase]})

        # deep copy the model
        if phase == 'val' and epoch_loss[phase] < best_loss:
            best_loss = epoch_loss[phase]
            best_audnet = copy.deepcopy(model.audionet.state_dict())
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    torch.save(best_audnet, 'best_audnet.pt')
    torch.save(best_model_wts, 'best_audcond.pt')
    # load best model weights
    model.load_state_dict(best_model_wts)
