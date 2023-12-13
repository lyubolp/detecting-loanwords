import sys
from itertools import accumulate


import numpy as np

from src.word_to_embedding import WordToEmbedding

from src.levenshtein import Levenshtein


def distance(word_1, word_2, w2e: WordToEmbedding, levenshtein: Levenshtein):
    # word_1_embedding = w2e.get_embedding(word_1)
    # word_2_embedding = w2e.get_embedding(word_2)

    return levenshtein.levenshtein(word_1, word_2)


def __accumulate_distance(x, y):
    return x + y


def __calculate_cell_value():
    pass


def levenshtein(word_1, word_2):
    m = len(word_1)
    n = len(word_2)

    dist = np.zeros((m + 1, n + 1))

    dist[1:, 0] = list(accumulate(word_1, func=__accumulate_distance, initial=0))[1:]
    dist[0, 1:] = list(accumulate(word_2, func=__accumulate_distance, initial=0))[1:]

    for j in range(1, n+1):
        deletion_cost = np.linalg.norm(word_2[j-1])  # Needs check
        for i in range(1, m+1):
            insertion_cost = np.linalg.norm(word_1[i-1])  # Needs check
            substitution_cost = 0

            distance = np.linalg.norm(word_1[i-1] - word_2[j-1])

            if distance > 10:
                substitution_cost = distance

            dist[i, j] = min(dist[i - 1, j] + deletion_cost,
                             dist[i, j - 1] + insertion_cost,
                             dist[i - 1, j - 1] + substitution_cost)

    return dist[len(word_1), len(word_2)]


def levenshtein_optimized(word_1, word_2, word_1_distances, word_2_distances):
    m = len(word_1)
    n = len(word_2)

    dist = np.zeros((m + 1, n + 1))

    dist[1:, 0] = list(accumulate(word_1_distances, func=lambda x, y: x + y, initial=0))[1:]
    dist[0, 1:] = list(accumulate(word_2_distances, func=lambda x, y: x + y, initial=0))[1:]

    for i in range(1, m+1):
        for j in range(1, n+1):
            dist[i, j] = calculate_cell(i, j, word_1, word_2, word_1_distances, word_2_distances[j-1],
                                        dist[i - 1, j], dist[i, j - 1], dist[i - 1, j - 1])

    return dist[len(word_1), len(word_2)]


def calculate_cell(i, j, word_1, word_2, word_1_distances, deletion_cost, left, up, diagonal):
    insertion_cost = word_1_distances[i-1]  # Needs check
    substitution_cost = 0

    distance = np.linalg.norm(word_1[i-1] - word_2[j-1])

    if distance > 10:
        substitution_cost = distance

    return min(left + deletion_cost, up + insertion_cost, diagonal + substitution_cost)
