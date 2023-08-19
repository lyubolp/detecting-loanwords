"""
Module containing the test cases for the utility functions
for the dataset
"""
import unittest

from transformer_model.src.dataset_utils import yield_tokens, sentence_to_tensor
from transformer_model.src.constants import SRC_LANGUAGE, TGT_LANGUAGE


def identity(item):
    """
    Simple identity function
    """
    return item


class DatasetUtilsTests(unittest.TestCase):
    """
    Unit tests for the DatasetUtils class
    """

    def test_01_yield_tokens(self):
        """
        Verify that the yield_tokens function works
        """
        # Arrange
        data_source_language = ['foo', 'foobar']
        data_target_language = ['bar', 'barfoo']

        data = list(zip(data_source_language, data_target_language))

        # Act
        iterator_source_language = yield_tokens(data, SRC_LANGUAGE, word_tokenizer=identity)
        iterator_target_language = yield_tokens(data, TGT_LANGUAGE, word_tokenizer=identity)

        # Assert
        for item in data_source_language:
            self.assertEqual(next(iterator_source_language), item)

        for item in data_target_language:
            self.assertEqual(next(iterator_target_language), item)

    def test_02_sentence_to_tensor(self):
        # Arrange
        # Act 
        # Assert
        pass
