import os
from typing import Dict, List

import pandas as pd
from nltk.tokenize import word_tokenize


class TranscriptionGeneration:
    """
    Class for generating transcription of a sentence
    """
    def __init__(self):
        self.letter_to_phoneme = {
            'а': lambda position, word: 'a' if self.__is_letter_emphasised(word, position) else 'ʌ',
            'б': lambda position, word: 'b',
            'в': lambda position, word: 'v',
            'г': lambda position, word: 'g',
            'д': lambda position, word: 'd',
            'е': lambda position, word: 'ɛ',
            'ж': lambda position, word: 'ʒ',
            'з': lambda position, word: 'z',
            'и': lambda position, word: 'i',
            'й': lambda position, word: 'j',
            'к': lambda position, word: 'k',
            'л': lambda position, word: 'l',
            'м': lambda position, word: 'm',
            'н': lambda position, word: 'n',
            'о': lambda position, word: 'ɔ' if self.__is_letter_emphasised(word, position) else 'o',
            'п': lambda position, word: 'p',
            'р': lambda position, word: 'r',
            'с': lambda position, word: 's',
            'т': lambda position, word: 't',
            'у': lambda position, word: 'u' if self.__is_letter_emphasised(word, position) else 'o',
            'ф': lambda position, word: 'f',
            'х': lambda position, word: 'x',
            'ц': lambda position, word: 'ts',
            'ч': lambda position, word: 'tʃ',
            'ш': lambda position, word: 'ʃ',
            'щ': lambda position, word: 'ʃt',
            'ъ': None,
            'ь': lambda position, word: 'j',
            'ю': lambda position, word: 'ju' if self.__is_letter_emphasised(word, position) else 'jo',
            'я': lambda position, word: 'ja' if self.__is_letter_emphasised(word, position) else 'jɐ'
        }
        self.__emphasis_generation = EmphasisGeneration(os.path.join('data', 'emphasis.csv'))

    def __is_letter_emphasised(self, word: str, letter: int) -> bool:
        return self.__emphasis_generation.is_letter_emphasised(word, letter)

    def __transcribe_word(self, word: str) -> str:
        """
        Transcribe a single word
        """
        word = word.lower()
        transcribed_letters = [self.letter_to_phoneme[letter](letter, word) for letter in word]
        return ''.join(transcribed_letters)

    def generate_transcription(self, sentence: str, pretty=False) -> str:
        """
        Generate transcription of a whole sentence
        """
        tokens = word_tokenize(sentence)
        transcription = [self.__transcribe_word(word) for word in tokens]
        return ' '.join(transcription)


class EmphasisGeneration:
    def __init__(self, emphasis_db_path: str):
        self.__path = emphasis_db_path
        self.__words = self.__load(self.__path)

    @staticmethod
    def __load(path) -> Dict[str, List[int]]:
        """
        Load emphasis data from csv file
        """
        emphasis_data = pd.read_csv(path)

        return emphasis_data.set_index('word').to_dict()['emphasis_indexes']

    def is_letter_emphasised(self, word: str, letter: int) -> bool:
        """
        Check if a letter is emphasised
        """
        if word not in self.__words:
            return False
        return letter in self.__words[word]


if __name__ == '__main__':
    input_sentence = input('Моля, въведете изречение:')
    transcription_generator = TranscriptionGeneration()
    print(transcription_generator.generate_transcription(input_sentence, pretty=True))
