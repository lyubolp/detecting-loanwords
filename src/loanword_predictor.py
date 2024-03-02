import json
import math

import torch


from src.classification_model import LoanwordClassifier


class LoanwordPredictor:
    def __init__(self, id_to_label_path, label_to_id_path, word_to_embedding, 
                 classifier_state_dict_path, classifier=LoanwordClassifier):
        with open(label_to_id_path) as fp:
            self.__label_to_id = json.load(fp)

        with open(id_to_label_path) as fp:
            id_to_label = json.load(fp)

        self.__id_to_label = {int(key): value for key, value in id_to_label.items()}
        self.__w2e = word_to_embedding

        self.__model = LoanwordClassifier(input_size=512, hidden_size=1024, output_size=len(self.__label_to_id))

        self.__model.load_state_dict(torch.load(classifier_state_dict_path, map_location=torch.device('cpu')))

    def predict(self, word):
        word_tensor = torch.Tensor(self.__w2e.get_embedding(word))

        word_tensor_shape = word_tensor.shape
        word_tensor = word_tensor.reshape((word_tensor_shape[0], 1, word_tensor_shape[1]))

        hidden = self.__model.init_hidden()

        for syllable_embedding in word_tensor:
            output, hidden = self.__model(syllable_embedding, hidden)

        # predicted_label, predicted_label_id = category_from_output(output, id_to_label)

        probabilities, label_ids = output.topk(5)

        return {self.__id_to_label[label_id.item()]: pow(math.e, probability.item()) 
                for probability, label_id in zip(probabilities[0], label_ids[0])}
