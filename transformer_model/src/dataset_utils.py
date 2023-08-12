"""
Some functions to help with the dataset
"""
import os
from typing import Iterable, Callable, List

import torch
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator

from constants import SRC_LANGUAGE, TGT_LANGUAGE, MAX_LENGTH, device

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str, word_tokenizer: Callable = word_tokenize) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield word_tokenizer(data_sample[language_index[language]])

# function to convert a sentence to a tensor
# Might be dead code
def sentence_to_tensor(content, target_size=MAX_LENGTH) -> torch.Tensor:
    # Add padding to the end of the sentence, so that the length is equal to target_size
    tensor = torch.tensor(content, dtype=torch.long, device=device).view(-1, 1)

    if tensor.size()[0] < target_size:
        padding = torch.zeros(target_size - tensor.size()[0], 1, dtype=torch.int32, device=device)
        tensor = torch.cat((tensor, padding), dim=0)

    return tensor

def load_files(transcriptions_root_directory: str) -> List[str]:
    [os.path.join(root, file)  for root, dirs, files in os.walk(transcriptions_root_directory) for file in files]


def build_vocab_transformation(dataset, language):
    tokens_generator = yield_tokens(dataset, language)
    vocab = build_vocab_from_iterator(tokens_generator,
                                            min_freq=1,
                                                    specials=const.special_symbols,
                                                    special_first=True)
