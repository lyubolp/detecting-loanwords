"""
Module that contains the TranscriptionDataset class
"""
import gc
from typing import Dict

import pandas as pd
from torch.utils.data import Dataset


class TranscriptionDataset(Dataset):
    """
    Class that represents the transcrition dataset
    """
    def __init__(self, files,
                 filepath_to_size_path='/mnt/d/Projects/masters-thesis/data/filepath_to_size.csv',
                 index_offset=0):
        self.__files = files
        self.__filepath_to_size = pd.read_csv(filepath_to_size_path).sort_values(by=['filepath'])

        self.__total_lines = self.__filepath_to_size['size'].sum()
        self.__index_to_file_index = self.__generate_index(self.__filepath_to_size) # <- This is the memory hog

        self.__filepath_to_start_and_end_index = self.__filepath_to_size.set_index('filepath').to_dict('index')
        self.__max_rows = max(self.__filepath_to_size['end_index'])

        self.__index_offset = index_offset

        # Free up some memory  
        del self.__filepath_to_size
        gc.collect()

        self.__last_file_path = ''
        self.__last_file = None


    def __len__(self):
        return self.__total_lines

    def __getitem__(self, idx):
        idx += self.__index_offset
        if idx >= self.__max_rows:
            raise IndexError

        current_file_index = self.__index_to_file_index[idx]
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

        return row[0], row[2]

    @staticmethod
    def __generate_index(filepath_to_size: pd.DataFrame) -> Dict[int, int]:
        result = {}

        for i, row in enumerate(filepath_to_size.iloc()):
            start_index, end_index = row['start_index'], row['end_index']

            for index in range(start_index, end_index):
                result[index] = i

        return result
