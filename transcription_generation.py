class TranscriptionGeneration:
    def __init__(self):
        self.letter_to_phoneme = {
            'е': lambda letter, word: 'ɛ',
            'и': lambda letter, word: 'i',
        }