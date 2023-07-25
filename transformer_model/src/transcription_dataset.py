"""
Module that contains the TranscriptionDataset class
"""
import pandas as pd

from itertools import accumulate
from torch.utils.data import Dataset
from typing import Dict


class TranscriptionDataset(Dataset):
    """
    Class that represents the transcrition dataset
    """
    def __init__(self, files, start_file=0,
                 filepath_to_size_path='/mnt/d/Projects/masters-thesis/data/filepath_to_size.csv'):
        self.__files = files
        self.__current_row = 0

        self.__filepath_to_size = pd.read_csv(filepath_to_size_path).sort_values(by=['filepath'])
        self.__filepath_to_size['end_index'] = list(accumulate(self.__filepath_to_size['size']))
        self.__filepath_to_size['start_index'] = [0] + list(self.__filepath_to_size['end_index'].iloc()[:-1])

        self.__index_to_file_index = self.__generate_index(self.__filepath_to_size)
        self.__filepath_to_start_and_end_index = self.__filepath_to_size.set_index('filepath').to_dict('index')
        self.__max_rows = max(self.__filepath_to_size['end_index'])

    def __len__(self):
        return len(self.__files)

    def __getitem__(self, idx):
        if idx >= self.__max_rows:
            raise IndexError

        current_file_path = self.__index_to_file_index[idx]
        current_file_path_start_index = self.__filepath_to_start_and_end_index[current_file_path]['start_index']

        idx = idx - current_file_path_start_index

        current_file = pd.read_csv(current_file_path)

        row = current_file.iloc[self.__current_row]

        return row[0], row[2]

    @staticmethod
    def __generate_index(filepath_to_size: pd.DataFrame) -> Dict[int, int]:
        result = {}

        for i, row in enumerate(filepath_to_size.iloc()):
            start_index, end_index = row['start_index'], row['end_index']

            for index in range(start_index, end_index):
                result[index] = i

        return result
