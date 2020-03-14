import os
import numpy as np
from sentiment_model import SentimentAnalyser
from variables import sentiment_path,sentiment_weights, seed
from util import get_evaluation_data, get_data

if __name__ == "__main__":
    Xtrain, Ytrain, Xtest , Ytest = get_data()
    model = SentimentAnalyser(Xtrain, Ytrain, Xtest , Ytest)
    model.tokenizing_data()
    if os.path.exists(sentiment_path) and os.path.exists(sentiment_weights):
        print("Loading existing model !!!")
        model.load_model()
    else:
        print("Training the model  and saving!!!")
        model.embedding_model()
        model.train_model()
        model.save_model()

    headlines, is_sarcastic = get_evaluation_data()
    model.predict(headlines, is_sarcastic)