import os
import sys
import math 
import time
import argparse

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


def train(train_loader, model, criterion, optimizer):

    epoch_loss = 0
    batch_num = 0
    for x, y in tqdm(train_loader, leave=False):

        optimizer.zero_grad()
        pred = model(x.to(DEVICE))
        pred_decode = decode(pred.cpu())
        pred_final = [correct_prediction(p) for p in pred_decode]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CaptchaOCR")
    parser.add_argument(
        "--resume",
        action = "store_true"
    )
    parser.add_argument(
        "--resume_checkpoint",
        type = str,
        default=None
    )
    args = parser.parse_args()
    
    # Define data paths
    train_data_path = '../data/train/'
    val_data_path = '../data/val/'
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
    data_letters = sorted(list(set(list(images))))
    print(f'There are {len(data_letters)} unique letters in the dataset: {data_letters}')

    if data_letters != letters:
        sys.exit('Dataset contains other letters')

    # Get batches of dataset
    train_dataset = CapchaDataset(train_data_path)
    val_dataset = CapchaDataset(val_data_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f'{len(train_loader)} batches in the train_loader')
    print(f'{len(val_loader)} batches in the test_loader')

    # Define model
    num_chars = len(vocabulary)
    rnn_hidden_size = 256

    model = CRNN(num_chars=num_chars, rnn_hidden_size=rnn_hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)

    if args.resume == True:
        checkpoint = torch.load(checkpoint_dir+args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        model.apply(initialize_weights)
        start_epoch = 1
    model = model.to(DEVICE)

    criterion = nn.CTCLoss(blank=0)

    # Define writer
    writer = SummaryWriter(log_dir)
    
    # Start Train & Evaluate
    best_valid_loss = float('inf')
    best_epoch = 0
    for epoch in tqdm(range(start_epoch, NUM_EPOCHS +1)):
        start_time = time.time()

        train_loss = train(train_loader, model, criterion, optimizer)
        val_loss = evaluate(val_loader, model, criterion)

        end_time = time.time()

        lr_scheduler.step(train_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/evaluate', val_loss, epoch)

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir+"epoch_%03d.pt"%(epoch))
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir+'crnn-best-model.pt')
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'\nEpoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')
        print(f'\tBest Epoch: {best_epoch}')



