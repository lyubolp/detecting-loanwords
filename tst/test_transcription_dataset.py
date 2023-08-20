"""
МModule
"""
import os
import unittest

from itertools import accumulate
from typing import List, Tuple

import pandas as pd

from src.transcription_dataset import TranscriptionDataset


class TranscriptDatasetTests(unittest.TestCase):
    """
    Unit tests for the TranscriptionDataset class
    """
    @classmethod
    def setUpClass(cls):
        """
        Set up the tests
        """

        # We need the following structure created

        # Files with some lines
        
        cls.__files_content = [
            [('apple', '', 'banana'), ('carrot', '', 'orange'), ('grape', '', 'melon')],
            [('house', '', 'tree'), ('river', '', 'ocean'), ('mount', '', 'hill'), ('city', '', 'town')],
            [('dog', '', 'cat'), ('bird', '', 'fish')],
            [('shirt', '', 'pants'), ('shoe', '', 'sock')],
            [('book', '', 'pen'), ('paper', '', 'pencil'), ('note', '', 'desk')]
        ]
        cls.__transcription_dir = '/tmp/transcription_dataset_tests'
        cls.__filenames = [f'file{i+1}.txt' for i in range(len(cls.__files_content))]
        cls.__files = [os.path.join(cls.__transcription_dir, filename) for filename in cls.__filenames]
        
        os.makedirs(cls.__transcription_dir, exist_ok=True)
        TranscriptDatasetTests.__create_files(cls.__files, cls.__files_content,
                                              cls.__transcription_dir)

        cls.__filepath_to_size_filename = os.path.join(cls.__transcription_dir,
                                                 'filepath_to_size.csv')
        filepath_to_size = TranscriptDatasetTests.__generate_filepath_to_size(cls.__files,
                                                                              cls.__files_content)
        filepath_to_size.to_csv(cls.__filepath_to_size_filename, index=False)


    # @classmethod
    # def tearDownClass(cls) -> None:

    #     TranscriptDatasetTests.__remove_files(cls.__transcription_dir, cls.__files)
    #     os.removedirs(cls.__transcription_dir)

    def test_01_no_start_no_end(self):
        # Arrange

        dataset = TranscriptionDataset(self.__files, self.__filepath_to_size_filename)
        expected = sum(self.__files_content, [])
        expected = [(item[0], item[2]) for item in expected]

        # Act
        actual = [dataset[i] for i in range(len(dataset))]

        # Assert
        self.assertEqual(expected, actual)

    def test_02_start_no_end(self):
        # Arrange
        start_index = 2

        dataset = TranscriptionDataset(self.__files, self.__filepath_to_size_filename,
                                       start_index=start_index)
        expected = sum(self.__files_content, [])
        expected = [(item[0], item[2]) for item in expected][start_index:]

        # Act
        actual = [dataset[i] for i in range(len(dataset))]

        # Assert
        self.assertEqual(expected, actual)

    def test_03_no_start_end(self):
        # Arrange
        end_index = 2

        dataset = TranscriptionDataset(self.__files, self.__filepath_to_size_filename,
                                       end_index=end_index)
        expected = sum(self.__files_content, [])
        expected = [(item[0], item[2]) for item in expected][:end_index]

        # Act
        actual = [dataset[i] for i in range(len(dataset))]

        # Assert
        self.assertEqual(expected, actual)


    def test_04_start_end_index(self):
        # Arrange
        start_index = 2
        end_index = 9

        dataset = TranscriptionDataset(self.__files, self.__filepath_to_size_filename,
                                       start_index=start_index, end_index=end_index)
        expected = sum(self.__files_content, [])
        expected = [(item[0], item[2]) for item in expected][start_index:end_index]

        # Act
        actual = [dataset[i] for i in range(len(dataset))]
    
        # Assert
        self.assertEqual(expected, actual)


    @staticmethod
    def __create_files(files: List[str], content: List[List[Tuple[str, str]]], directory_path: str):
        for filepath, file_content in zip(files, content):
            with open(filepath, 'w+', encoding='utf-8') as filepointer:
                filepointer.writelines('word,transcription,translation\n')
                for pair in file_content:
                    filepointer.writelines(','.join(pair) + '\n')

    @staticmethod
    def __remove_files(directory_path: str, files: List[str]):
        for filename in files:
            filepath = os.path.join(directory_path, filename)
            os.remove(filepath)

    @staticmethod
    def __generate_filepath_to_size(files: List[str], contents: List[List[Tuple[str, str]]]):
        filepath_to_size = pd.DataFrame()
        filepath_to_size['filepath'] = files
        filepath_to_size['size'] = [len(content) for content in contents]
        filepath_to_size['end_index'] = list(accumulate(filepath_to_size['size']))
        filepath_to_size['start_index'] = [0] + list(filepath_to_size['end_index'].iloc()[:-1])

        return filepath_to_size