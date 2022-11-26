import torch
from torch.utils.data import Dataset
from LandmarkDataset import FaceLandmarksDataset
from LandmarkModels import LandmarkEncoder, LandmarkDecoder, LandmarkAutoencoder
import copy
import time
import wandb
wandb.init(project="Landmark-Disentangler")

batch_size = 256
num_epochs = 10
switch_factor = 0.8
weight_decay = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.config={"batch_size":batch_size,
              "epochs": num_epochs,
              "switch_factor": switch_factor,
              "weight_decay": weight_decay,
              }

train_landmarks_dataset = FaceLandmarksDataset(csv_file='/scratch/tan/train_landmarks.csv')
val_landmarks_dataset = FaceLandmarksDataset(csv_file='/scratch/tan/val_landmarks.csv')

dataset_sizes = {'train': len(train_landmarks_dataset), 'val':val_landmarks_dataset}
train_dataloader = torch.utils.data.DataLoader(train_landmarks_dataset, batch_size=256, shuffle=True, num_workers=10)
val_dataloader = torch.utils.data.DataLoader(val_landmarks_dataset, batch_size=256, shuffle=True, num_workers=10)

dataloaders = {'train':train_dataloader, 'val': val_dataloader}

model = LandmarkAutoencoder(switch_factor)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
since = time.time()
for epoch in range(num_epochs):
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
        for landmarks_a, landmarks_b in dataloaders[phase]:
            landmarks_a = landmarks_a.to(device)
            landmarks_b = landmarks_b.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                loss_a, loss_b, loss = model(landmarks_a, landmarks_b)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss * landmarks_a.size(0)

        epoch_loss = running_loss / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f}')
        wandb.log({phase+"_loss": epoch_loss})

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    torch.save(best_model_wts, 'best_autoencoder.pt')
    # load best model weights
    model.load_state_dict(best_model_wts)
