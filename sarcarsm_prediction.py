import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
from sentiment_model import SentimentAnalyser
from variables import sentiment_path,sentiment_weights, seed
from util import get_evaluation_data, get_data

if __name__ == "__main__":
    model = SentimentAnalyser()
    model.run()