import json

import pandas as pd
import torch

from torch import nn

from src.word_to_embedding import WordToEmbedding

RANDOM_STATE = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoanwordClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.__hidden_size = hidden_size

        self.__i2h = nn.Linear(input_size + self.__hidden_size, self.__hidden_size).to(device)
        self.__h2o = nn.Linear(hidden_size, output_size).to(device)
        self.__softmax = nn.LogSoftmax(dim=1).to(device)

    def forward(self, input_tensor, hidden):
        combined = torch.cat((input_tensor.to(device), hidden.to(device)), 1).to(device)
        hidden = self.__i2h(combined).to(device)
        output = self.__h2o(hidden).to(device)
        output = self.__softmax(output).to(device)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.__hidden_size)