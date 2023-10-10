"""
Some functions to help with the dataset
"""
import os
from typing import Iterable, Callable, List, Generator

import torch
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator

import src.constants as const
from src.syllable_splitter import split_word

vowels_transcription = ['a', 'ʌ', 'ɤ̞',  'ɐ', 'ɔ', 'o', 'u', 'ɛ', 'i']

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str, word_tokenizer: Callable = word_tokenize) -> Generator:
    language_index = {const.SRC_LANGUAGE: 0, const.TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield data_sample[language_index[language]]

# function to convert a sentence to a tensor
# Might be dead code
def sentence_to_tensor(content, target_size=const.MAX_LENGTH) -> torch.Tensor:
    # Add padding to the end of the sentence, so that the length is equal to target_size
    tensor = torch.tensor(content, dtype=torch.long, device=const.device).view(-1, 1)

    if tensor.size()[0] < target_size:
        padding = torch.zeros(target_size - tensor.size()[0], 1, dtype=torch.int32, device=const.device)
        tensor = torch.cat((tensor, padding), dim=0)

    return tensor

def load_files(transcriptions_root_directory: str) -> List[str]:
    return [os.path.join(root, file)  for root, dirs, files in os.walk(transcriptions_root_directory) for file in files]


def build_vocab_transformation(dataset, language):
    tokens_generator = yield_tokens(dataset, language)
    vocab = build_vocab_from_iterator(tokens_generator, min_freq=1, specials=const.special_symbols,
                                      special_first=True)

    # Set ``const.UNK_IDX`` as the default index. This index is returned when the token is not found.
    # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
    vocab.set_default_index(const.UNK_IDX)

    return vocab

def tokenize_source(sentence) -> list[str]:
    return __tokenize(sentence)

def tokenize_target(sentence) -> list[str]:
    return __tokenize(sentence, vowels_transcription)

def __tokenize(sentence, volews=None) -> list[str]:
    word_tokenized = word_tokenize(sentence)
    syllables_tokenized = [split_word(word, volews) for word in word_tokenized]
    flattened: list[str] = sum(syllables_tokenized, [])

    return flattened

def identity(*args):
    return args