import os
import json
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentiment_model import SentimentAnalyser
import logging
logging.getLogger('tensorflow').disabled = True
from variables import *
from util import*

from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)

model = SentimentAnalyser()
model.run()


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    input_text = str(message['input_text'])
    is_sacastic = model.predicts(input_text)

    response = {
            'is_sarcastic': is_sacastic
    }
    return jsonify(response)

