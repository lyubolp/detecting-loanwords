from itertools import accumulate

import numpy as np

from src.word_to_embedding import get_embedding


def distance(word_1, word_2):
    word_1_embedding = get_embedding(word_1)
    word_2_embedding = get_embedding(word_2)

    return __levenshtein(word_1_embedding, word_2_embedding)


def __levenshtein(word_1, word_2):
    m = len(word_1)
    n = len(word_2)

    dist = np.zeros((m + 1, n + 1))

    dist[1:, 0] = list(accumulate(word_1, func=lambda x, y: x + np.linalg.norm(y), initial=0))[1:]
    dist[0, 1:] = list(accumulate(word_2, func=lambda x, y: x + np.linalg.norm(y), initial=0))[1:]

    for j in range(1, n+1):
        for i in range(1, m+1):
            deletion_cost = np.linalg.norm(word_2[j-1])  # Needs check
            insertion_cost = np.linalg.norm(word_1[i-1])  # Needs check
            substitution_cost = 0

            distance = np.linalg.norm(word_1[i-1] - word_2[j-1])
            if distance > 10:
                substitution_cost = distance

            dist[i, j] = min(dist[i - 1, j] + deletion_cost,
                             dist[i, j - 1] + insertion_cost,
                             dist[i - 1, j - 1] + substitution_cost)

    return dist[len(word_1), len(word_2)]
