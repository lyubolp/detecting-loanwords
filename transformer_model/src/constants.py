"""
Module containing the constants
"""
import torch


MAX_LENGTH = 1000
SOS_TOKEN = 2648747
EOS_TOKEN = 2648748
TEACHER_FORCING_RATIO = 0.5
LEARNING_RATE = 0.01
HIDDEN_SIZE = 256

TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42

SRC_LANGUAGE = 'src'
TGT_LANGUAGE = 'tgt'

vocab_transform = {}

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
