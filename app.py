import json
import requests

from flask import Flask, make_response, request, render_template, redirect, url_for
from nltk.tokenize import word_tokenize


API_URL = 'http://127.0.0.1:5000'

app = Flask(__name__)


def tokenize_sentence(sentence: str) -> list[str]:
    words = word_tokenize(sentence)
    return [word.lower() for word in words]


def loanwords_analysis(sentence: str) -> list[tuple[str, dict[str, float]]]:
    words = tokenize_sentence(sentence)

    predictions = [(word, predictor.predict(word)) for word in words]

    predictions = [(word, probabilities)
                   for word, probabilities in predictions
                   if 'bg' not in probabilities or probabilities['bg'] < 0.8]
    return predictions


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

    data = {
        'sentence': sentence
    }
    json_data = json.dumps(data)

    response = requests.post(API_URL + '/api/analyze', data=json_data, headers={'Content-Type': 'application/json'})

    if not response:
        return make_response("Error when communicating to the API", 501)

    print(response.json())
    words = tokenize_sentence(sentence)
    result = response.json()

    return results(words, result)


def results(words=None, analysis=None):
    print(words)
    print(analysis)
    analysis_dict = {word: results for word, results in analysis}
    return render_template('results.html', words=words, analysis=analysis_dict)


@app.route("/status")
def status():
    return make_response("OK", 200)
