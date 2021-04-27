import os
import math 
import time
import argparse

from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from dataset import CapchaDataset, remove_file_extension
from config import *
from model import CRNN
from utils.util import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_count = mp.cpu_count()


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
    
    parser = argparse.ArgumentParser(description="CaptchaOCR")
    parser.add_argument(
        "--sc",
        action = "store_true"
    )
    parser.add_argument(
        "--kab",
        action = "store_true"
    )
    parser.add_argument(
        "--checkpoint",
        type = str
    )
    args = parser.parse_args()

    # Define data paths
    if args.kab:
        test_data_path = '../data/test_kab/'
        csv_path = './result_kab.csv'
    if args.sc:
        test_data_path = '../data/test/'
        csv_path = './result.csv' 
    else:
        print("assign test dataset")
        sys.exit()

    # Define character maps
    vocabulary = ["-"] + letters
    idx2char = {k:v for k,v in enumerate(vocabulary, start=0)}
    char2idx = {v:k for k,v in idx2char.items()}
    print(len(vocabulary))
    print(idx2char)
    print(char2idx)

    # Get batches of dataset
    test_dataset = CapchaDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f'{len(test_loader)} batches in the test_loader')

    # Define model
    num_chars = len(vocabulary)
    rnn_hidden_size = 256

    model = CRNN(num_chars=num_chars, rnn_hidden_size=rnn_hidden_size)
    checkpoint = torch.load('checkpoints/'+ args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    criterion = nn.CTCLoss(blank=0)
    
    # Start Test
    start_time = time.time()
 
    test_loss, result = test(test_loader, model, criterion)
    acc = accuracy_score(result['Actual'], result['Prediction'])
    result.to_csv(csv_path)
    end_time = time.time()
    
    test_mins, test_secs = epoch_time(start_time, end_time)
    print(f'\nTest Time: {test_mins}m {test_secs}s')
    print(f'\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
    print(f'\tAccuracy: {acc:.3f}')


