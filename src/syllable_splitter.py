"""
Module that contains the function that splits a word into syllables.
"""
__all__ = ['split_word']

vowels = ['а', 'ъ', 'о', 'у', 'е', 'и']

def get_vowels_count(word: str) -> int:
    """
    Returns the amount of vowels in a word.
    :param word: The word to check.
    :return: The amount of vowels in the word.
    """
    return len([x for x in word if x in vowels])

def get_first_vowel_location(word: str) -> int:
    """
    Returns the index of the first vowel in a word.
    :param word: The word to check.
    :return: The index of the first vowel in the word.
    """
    for index, letter in enumerate(word):
        if letter in vowels:
            return index
    return -1

def split_word(word: str) -> list[str]:
    """
    Splits a word into syllables.
    :param word: The word to split.
    :return: A list of syllables.
    """
    volews_count = get_vowels_count(word)

    result = []

    while volews_count > 1:
        first_vowel_location = get_first_vowel_location(word)
        second_vowel_location = get_first_vowel_location(word[first_vowel_location + 1:]) + first_vowel_location + 1

        amount_of_consonants = second_vowel_location - first_vowel_location - 1

        if amount_of_consonants == 1:
            # 65.1
            cut_index = first_vowel_location + 1
        elif amount_of_consonants >= 2:
            if amount_of_consonants == 2 and word[first_vowel_location + 1] == word[first_vowel_location + 2]:
                # 65.2.1
                cut_index = first_vowel_location + 2
            else:
                # 65.2
                cut_index = first_vowel_location + (amount_of_consonants // 2) + 1
        elif amount_of_consonants == 0:
            # 65.3
            cut_index = first_vowel_location + 1
        
        result.append(word[:cut_index])
        word = word[cut_index:]
        volews_count = get_vowels_count(word)
    
    result.append(word)

    return result