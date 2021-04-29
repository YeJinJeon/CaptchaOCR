BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
CLIP_NORM = 5

letters = ['2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'm', 'n', 'p', 'r', 'w', 'x', 'y']
vocabulary = ["-"] + letters
idx2char = {k:v for k,v in enumerate(vocabulary, start=0)}
char2idx = {v:k for k,v in idx2char.items()}