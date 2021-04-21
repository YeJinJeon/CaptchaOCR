import os
import math 
import time

from tqdm import tqdm
import multiprocessing as mp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CapchaDataset, remove_file_extension
from config import *
from model import CRNN
from utils.util import *

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


def compute_loss(gtruth, pred, criterion):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    predicted_capchas = F.log_softmax(pred, 2)
    predicted_capchas_lens = torch.full(size=(predicted_capchas.size(1),), 
                                       fill_value=predicted_capchas.size(0), 
                                       dtype=torch.int32).to(DEVICE)

    gtruth_capchas, gtruth_capchas_lens = encode(gtruth)
    loss = criterion(predicted_capchas, gtruth_capchas, predicted_capchas_lens, gtruth_capchas_lens)

    return loss

def train(train_loader, model, criterion, optimizer):

    epoch_loss = 0
    batch_num = 0
    for x, y in tqdm(train_loader, leave=False):

        optimizer.zero_grad()
        pred = model(x.to(DEVICE))
        pred_decode = decode(pred.cpu())
        pred_final = [correct_prediction(p) for p in pred_decode]
        print(f'\npred: {pred_final}')
        print(f'target: {y}')
        loss = compute_loss(y, pred, criterion)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        epoch_loss += loss.item()
        batch_num += 1

    return epoch_loss / batch_num 


def evaluate(val_loader, model, criterion):

    epoch_loss = 0
    batch_num = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, leave=False):

            pred = model(x.to(DEVICE))

            loss = compute_loss(y, pred, criterion)

            epoch_loss += loss.item()
            batch_num += 1

    return epoch_loss / batch_num 


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    
    # Define data paths
    train_data_path = '/mnt/c/Users/samsung/tanker/data/simplecaptcha/train/'
    val_data_path = '/mnt/c/Users/samsung/tanker/data/simplecaptcha/val/'
    log_dir = './logs/'
    checkpoint_dir = './checkpoints/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

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
    val_dataset = CapchaDataset(val_data_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=cpu_count, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=cpu_count, shuffle=False)

    print(f'{len(train_loader)} batches in the train_loader')
    print(f'{len(val_loader)} batches in the test_loader')

    # Define model
    num_chars = len(vocabulary)
    rnn_hidden_size = 256

    model = CRNN(num_chars=num_chars, rnn_hidden_size=rnn_hidden_size)
    model.apply(initialize_weights)
    model = model.to(DEVICE)

    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)

    # Define writer
    writer = SummaryWriter(log_dir)
    
    # Start Train & Evaluate
    best_valid_loss = float('inf')
    best_epoch = 0
    for epoch in tqdm(range(1, NUM_EPOCHS +1)):
        start_time = time.time()

        train_loss = train(train_loader, model, criterion, optimizer)
        val_loss = evaluate(val_loader, model, criterion)

        end_time = time.time()

        lr_scheduler.step(train_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/evaluate', val_loss, epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoint_dir+"epoch_%03d_loss_%.06f.pt"%(epoch, val_loss))
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_dir+'crnn-best-model.pt')
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'\nEpoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')
        print(f'\tBest Epoch: {best_epoch}')



