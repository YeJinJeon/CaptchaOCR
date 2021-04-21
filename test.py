import os
import math 
import time

from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

from dataset import CapchaDataset, remove_file_extension
from config import *
from model import CRNN
from train import compute_loss, epoch_time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_count = mp.cpu_count()

def decode(labels):
    tokens = F.softmax(labels, 2).argmax(2)
    tokens = tokens.numpy().T
    capchas = []
    
    for token in tokens:
        chars = [idx2char[idx] for idx in token]
        capcha = ''.join(chars)
        capchas.append(capcha)
    return capchas


def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)


def correct_prediction(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word


def test(test_loader, model, criterion):

    test_loss = 0
    batch_num = 0

    results_test = pd.DataFrame(columns=['Actual', 'Prediction'])

    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):

            pred = model(x.to(DEVICE))
    
            loss = compute_loss(y, pred, criterion)

            test_loss += loss.item()
            batch_num += 1

            pred = decode(pred.cpu())
            df = pd.DataFrame(columns=['Actual', 'Prediction'])
            df['Actual'] = y
            df['Prediction'] = [correct_prediction(p) for p in pred]
            results_test = pd.concat([results_test, df])
            results_test = pd.concat([results_test, df])

        results_test = results_test.reset_index(drop=True)
        results_test = results_test.reset_index(drop=True)

    return (test_loss / batch_num), results_test


if __name__ == "__main__":
    
    # Define data paths
    test_data_path = '/mnt/c/Users/samsung/tanker/data/simplecaptcha/test/'

    # Define character maps
    letters = ['2', '3', '4', '5', '6', '7', '8', 
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'm', 'n', 'p', 'r', 'w', 'x', 'y']
    vocabulary = ["-"] + letters
    idx2char = {k:v for k,v in enumerate(vocabulary, start=0)}
    char2idx = {v:k for k,v in idx2char.items()}
    print(len(vocabulary))
    print(idx2char)
    print(char2idx)

    # Get batches of dataset
    test_dataset = CapchaDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=cpu_count, shuffle=True)
    print(f'{len(test_loader)} batches in the test_loader')

    # Define model
    num_chars = len(vocabulary)
    rnn_hidden_size = 256

    model = CRNN(num_chars=num_chars, rnn_hidden_size=rnn_hidden_size)
    model.load_state_dict(torch.load('checkpoints/crnn-best-model.pt'))
    model = model.to(DEVICE)

    criterion = nn.CTCLoss(blank=0)
    
    # Start Test
    start_time = time.time()

    test_loss, result = test(test_loader, model, criterion)
    acc = accuracy_score(result['Actual'], result['Prediction'])
    result.to_csv('./result.csv')
    end_time = time.time()
    
    test_mins, test_secs = epoch_time(start_time, end_time)
    print(f'\nTest Time: {test_mins}m {test_secs}s')
    print(f'\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
    print(f'\tAccuracy: {acc:.3f}')


