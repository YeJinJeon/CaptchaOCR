import multiprocessing as mp

import torch
import torch.nn.functional as F

from config import * 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def decode(labels):
    tokens = F.softmax(labels, 2).argmax(2)
    tokens = tokens.numpy().T
    capchas = []
    
    for token in tokens:
        chars = [idx2char[idx] for idx in token]
        capcha = ''.join(chars)
        capchas.append(capcha)
    return capchas


def encode(labels):
    lens = [len(label) for label in labels]
    lens = torch.IntTensor(lens)
    
    labels_string = ''.join(labels)
    targets = [char2idx[char] for char in labels_string]
    targets = torch.IntTensor(targets).to(DEVICE)
    
    return (targets, lens)


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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs