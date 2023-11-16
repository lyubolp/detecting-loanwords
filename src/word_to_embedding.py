# Third party imports
import numpy as np
import pandas as pd
import torch
import seaborn as sns

from sklearn.cluster import KMeans

import src.constants as const
from src.dataset_utils import load_files, build_vocab_transformation, tokenize_source, tokenize_target
from src.transcription_dataset_single_word import TranscriptionDataset
from src.transformer_model import Seq2SeqTransformer
from src.syllable_splitter import split_word


torch.manual_seed(0)

SRC_VOCAB_SIZE = 5187  # Hardcoded for now
TGT_VOCAB_SIZE = 3387  # Hardcoded for now
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
transformer = transformer.to(const.device)


def __load_model():
    pass


def get_embedding(word: str):
    model = __load_model()
    word = word.lower()
    model.eval()
    src = text_transform_src(word).view(-1, 1).to(const.device)

    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    
    embedding = model.encode(src.to(const.device), src_mask.to(const.device))[1:-1]

    return embedding[:, 0, :].cpu().detach().numpy()
    # return (sum(embedding) / len(embedding))[0].cpu().detach().numpy()