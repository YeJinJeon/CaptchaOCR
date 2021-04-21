import os

from tqdm.notebook import tqdm
import multiprocessing as mp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CapchaDataset, remove_file_extension
from config import *
from model import CRNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_count = mp.cpu_count()

def initialize_weights(m):
    class_name = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(train_loader, model, criterion, optimizer):
    epoch_losses = []
    iteration_losses = []
    num_updates_epochs = []
    
    for epoch in tqdm(range(1, NUM_EPOCHS +1)):
        epoch_loss_list = []
        num_updates_epoch = 0
    
        for x, y in tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            pred = crnn(x.to(DEVICE))
            loss = compute_loss(y, pred)
            iteration_loss = loss.item()
            
            if np.isnan(iteration_loss) or np.isinf(iteration_loss):
                continue
            
            num_updates_epoch += 1
            iteration_losses.append(iteration_loss)
            epoch_loss_list.append(iteration_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(crnn.parameters(), CLIP_NORM)
            optimizer.step()
            
        epoch_loss = np.mean(epoch_loss_list)
        print(f'Epoch {epoch}:    Loss {loss}     Num_updates {num_updates_epoch}')
    
    epoch_losses.append(epoch_loss)
    num_updates_epochs.append(num_updates_epoch)
    # lr_scheduler.step(epoch_loss)


if __name__ == "__main__":
    
    # Define data paths
    train_data_path = '/mnt/c/Users/samsung/tanker/data/simplecaptcha/train/'
    val_data_path = '/mnt/c/Users/samsung/tanker/data/simplecaptcha/val/'

    # Define character maps
    captcha_images = []
    captcha_images.extend(os.listdir(train_data_path))
    captcha_images.extend(os.listdir(val_data_path))
    print(f'Total images {len(captcha_images)}')

    images = [remove_file_extension(image) for image in captcha_images]
    images = "".join(images)
    letters = sorted(list(set(list(images))))
    print(f'There are {len(letters)} unique letters in the dataset: {letters}')

    vocabulary = ["-"] + letters
    idx2char = {k:v for k,v in enumerate(vocabulary, start=0)}
    char2idx = {v:k for k,v in idx2char.items()}
    print(len(vocabulary))
    print(idx2char)
    print(char2idx)

    # Get batches of dataset
    train_dataset = CapchaDataset(train_data_path)
    test_dataset = CapchaDataset(val_data_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=cpu_count, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=cpu_count, shuffle=False)

    print(f'{len(train_loader)} batches in the train_loader')
    print(f'{len(test_loader)} batches in the test_loader')

    # Define model
    num_chars = len(vocabulary)
    rnn_hidden_size = 256

    crnn = CRNN(num_chars=num_chars, rnn_hidden_size=rnn_hidden_size)
    crnn.apply(initialize_weights)
    crnn = crnn.to(DEVICE)

    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(crnn.parameters(), lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)
    


