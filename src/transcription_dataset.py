"""
Module that contains the TranscriptionDataset class
"""
import bisect
import gc
from typing import Dict

import pandas as pd
from torch.utils.data import Dataset

from src.syllable_splitter import split_word
from src.dataset_utils import identity


vowels_transcription = ['a', 'ʌ', 'ɤ̞',  'ɐ', 'ɔ', 'o', 'u', 'ɛ', 'i']


class TranscriptionDataset(Dataset):
    """
    Class that represents the transcrition dataset
    """
    def __init__(self, files,
                 tokenization_src=None,
                 tokenization_tgt=None,
                 filepath_to_size_path='/mnt/d/Projects/masters-thesis/data/filepath_to_size.csv',
                 start_index=0, end_index=-1):
        self.__files = files
        self.__filepath_to_size = pd.read_csv(filepath_to_size_path).sort_values(by=['filepath'])

        self.__index_to_file_index = self.__generate_index(self.__filepath_to_size) # <- This is the memory hog
        self.__files_start_indexes = list(self.__index_to_file_index.keys())

        self.__filepath_to_start_and_end_index = self.__filepath_to_size.set_index('filepath').to_dict('index')
        self.__max_rows: int = max(self.__filepath_to_size['end_index'])

        self.__start_index = start_index
        self.__end_index = end_index if end_index != -1 else self.__max_rows
        self.__length = self.__end_index - self.__start_index

        # Free up some memory
        del self.__filepath_to_size
        gc.collect()

        self.__last_file_path = ''
        self.__last_file = None

        self.__tokenization_src = tokenization_src if tokenization_src is not None else identity
        self.__tokenization_tgt = tokenization_tgt if tokenization_tgt is not None else identity


    def __len__(self):
        return self.__length

    def __getitem__(self, idx):
        if idx >= self.__length:
            raise IndexError

        idx += self.__start_index
        # need to find the start index
        # from the start index, need to find which file index that is
        # from the file index, the file needs to be found

        start_index = self.__files_start_indexes[bisect.bisect(self.__files_start_indexes, idx) - 1]
        current_file_index = self.__index_to_file_index[start_index]
        current_file_path = self.__files[current_file_index]
        current_file_path_start_index = self.__filepath_to_start_and_end_index[current_file_path]['start_index']

        idx = idx - current_file_path_start_index

        if current_file_path != self.__last_file_path:
            current_file = pd.read_csv(current_file_path)

            self.__last_file_path = current_file_path
            self.__last_file = current_file
        else:
            current_file = self.__last_file

        row = current_file.iloc[idx]

        if isinstance(row[0], str):
            sentence = row[0]
        else:
            sentence = ''

        if isinstance(row[2], str):
            transcription = row[2]
        else:
            transcription = ''

        return self.__tokenization_src(sentence), self.__tokenization_tgt(transcription)

    @staticmethod
    def __generate_index(filepath_to_size: pd.DataFrame) -> Dict[int, int]:
        result = {}

        for i, row in enumerate(filepath_to_size.iloc()):
            start_index = row['start_index']
            result[start_index] = i

        return result
