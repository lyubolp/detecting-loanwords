import json

from flask import Flask, make_response, request, render_template, redirect, url_for
from nltk.tokenize import word_tokenize

from src.loanword_predictor import LoanwordPredictor
from src.word_to_embedding import WordToEmbedding


w2e = WordToEmbedding()
predictor = LoanwordPredictor(id_to_label_path='models/id-to-label-2024-02-06-1024hidden-10epochs.json',
                              label_to_id_path='models/label-to-id-2024-02-06-1024hidden-10epochs.json',
                              word_to_embedding=w2e,
                              classifier_state_dict_path='models/classifier-2024-02-06-1024hidden-10epochs.pth')
app = Flask(__name__)


def tokenize_sentence(sentence: str) -> list[str]:
    words = word_tokenize(sentence)
    return [word.lower() for word in words]


def loanwords_analysis(sentence: str) -> list[tuple[str, dict[str, float]]]:
    words = tokenize_sentence(sentence)

    return [(word, probabilities) for word in words if (probabilities := predictor.predict(word))['bg'] < 0.8]


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/analyze", methods=['GET', 'POST'])
def analyze():
    request_body = request.form

    print(request_body)
    if 'sentence' not in request_body:
        return make_response('Please pass sentence to the API', 400)

    sentence = request_body['sentence']

    words = tokenize_sentence(sentence)
    result = loanwords_analysis(sentence)

    return results(words, result)


@app.route("/api/analyze", methods=['POST'])
def api_analyze():
    request_body = request.json

    if 'sentence' not in request_body:
        return make_response('Please pass sentence to the API', 400)

    sentence = request_body['sentence']

    result = loanwords_analysis(sentence)
    return json.dumps(result, ensure_ascii=False)


def results(words=None, analysis=None):
    print(words)
    print(analysis)
    analysis_dict = {word: results for word, results in analysis}
    return render_template('results.html', words=words, analysis=analysis_dict)


@app.route("/status")
def status():
    return make_response("OK", 200)
