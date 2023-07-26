"""
МModule
"""
import os
import unittest

import pandas as pd

from transformer_model.src.transcription_dataset import TranscriptionDataset


class TranscriptDatasetTests(unittest.TestCase):
    """
    Unit tests for the TranscriptionDataset class
    """
    @classmethod
    def setUpClass(cls):
        cls.__transcription_dir = '/mnt/d/Projects/masters-thesis/data/transcriptions'
        cls.__files = [os.path.join(root, file) for root, dirs, files in os.walk(cls.__transcription_dir) for file in files]

    def test_01_check_first_index(self):
        """
        Check if the index 0 of the dataset is correct
        """
        # Arrange
        target_index = 0

        expected_file_path = self.__files[0]
        expected_file = pd.read_csv(expected_file_path)

        expected_input = expected_file.iloc[target_index][0]
        expected_output = expected_file.iloc[target_index][2]

        dataset = TranscriptionDataset(self.__files)

        # Act
        actual_input, actual_output = dataset[target_index]

        # Assert
        self.assertEqual(expected_input, actual_input)
        self.assertEqual(expected_output, actual_output)

    def test_02_check_file_switch(self):
        """
        Verify that the file switch logic works
        """
        # Arrange
        with open(self.__files[0], encoding='utf-8') as file_pointer:
            rows_in_first_file = len(file_pointer.readlines()) - 1

        target_index = rows_in_first_file + 1

        expected_file_path = self.__files[1]
        expected_file = pd.read_csv(expected_file_path)

        expected_input = expected_file.iloc[0][0]
        expected_output = expected_file.iloc[0][2]

        dataset = TranscriptionDataset(self.__files)

        # Act
        actual_input, actual_output = dataset[target_index]

        # Assert
        self.assertEqual(expected_input, actual_input)
        self.assertEqual(expected_output, actual_output)
