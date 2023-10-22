"""
Module containing the constants
"""
from typing import Dict

import torch
from torchtext.vocab import Vocab

MAX_LENGTH: int = 1000
SOS_TOKEN: int = 2648747
EOS_TOKEN: int = 2648748
TEACHER_FORCING_RATIO: float = 0.5
LEARNING_RATE: float = 0.01
HIDDEN_SIZE: int = 256

TRAIN_TEST_SPLIT: float = 0.8
TRAIN_VALIDATION_SPLIT: float = 0.1
RANDOM_SEED: int = 42

SRC_LANGUAGE: str = 'src'
TGT_LANGUAGE: str = 'tgt'

vocab_transform: Dict[str, Vocab] = {}

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
