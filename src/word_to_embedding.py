# Third party imports
import torch

import src.constants as const
from src.dataset_utils import build_vocab_transformation, tokenize_source
from src.transcription_dataset_single_word import TranscriptionDataset
from src.transformer_model import Seq2SeqTransformer
from src.training_utils import tensor_transform, sequential_transforms
from src.syllable_splitter import split_word


torch.manual_seed(0)

SRC_VOCAB_SIZE = 5187  # Hardcoded for now
TGT_VOCAB_SIZE = 3387  # Hardcoded for now
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


class WordToEmbedding:
    def __init__(self):
        self.__model = self.__load_model('models/transformer-single-word-2023-11-10-606102-25.pth')
        self.__dataset = self.__load_dataset('/mnt/d/Projects/masters-thesis/data/single_words.txt')
        self.__text_transform_src = self.__get_text_transform(self.__dataset)

    @staticmethod
    def __load_model(state_dict_path: str) -> Seq2SeqTransformer:
        transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                         NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
        transformer = transformer.to(const.device)
        transformer.load_state_dict(torch.load(state_dict_path))

        return transformer

    @staticmethod
    def __load_dataset(words_filepath: str) -> TranscriptionDataset:
        vowels_transcription = ['a', 'ʌ', 'ɤ̞',  'ɐ', 'ɔ', 'o', 'u', 'ɛ', 'i']

        with open(words_filepath, 'r') as f:
            words = f.readlines()

        amount_of_words = len(words)

        sentences_to_use = amount_of_words

        train_split = int(const.TRAIN_TEST_SPLIT * sentences_to_use)

        train_dataset = TranscriptionDataset(words_filepath, tokenization_src=split_word,
                                             tokenization_tgt=lambda x: split_word(x, vowels_transcription),
                                             start_index=0, end_index=train_split)

        return train_dataset

    @staticmethod
    def __get_text_transform(dataset: TranscriptionDataset):
        vocab_transform = build_vocab_transformation(dataset, const.SRC_LANGUAGE)

        text_transform_src = sequential_transforms(tokenize_source,
                                                   vocab_transform, tensor_transform)

        return text_transform_src

    def get_embedding(self, word: str):
        word = word.lower()
        self.__model.eval()
        src = self.__text_transform_src(word).view(-1, 1).to(const.device)

        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        embedding = self.__model.encode(src.to(const.device), src_mask.to(const.device))[1:-1]

        return embedding[:, 0, :].cpu().detach().numpy()
