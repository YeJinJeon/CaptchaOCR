import torch
import torch.nn.functional as F
from config import * 

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
    targets = torch.IntTensor(targets)
    
    return (targets, lens)


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