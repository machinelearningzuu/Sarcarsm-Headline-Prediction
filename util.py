import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from variables import data_path, filepath, train_path, test_path

def get_data():
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        if not os.path.exists(filepath):
            print("Load data from {}".format(data_path))
            Ydata = []
            Xdata = []
            preprocessed_Xdata = []
            for f in open(data_path,'r'):
                data = json.loads(f)
                if (str(data['headline']) is not None) and (int(data['is_sarcastic']) is not None):
                    preprocessed_text = preprocessed_headline(str(data['headline']))
                    if preprocessed_text is not None:
                        Ydata.append(int(data['is_sarcastic']))
                        Xdata.append(str(data['headline']))
                        preprocessed_Xdata.append(preprocessed_text)
            Xdata = np.array(Xdata)
            Ydata = np.array(Ydata)
            preprocessed_Xdata = np.array(preprocessed_Xdata)
            df = pd.DataFrame({'is_sarcastic': Ydata, 'headlines': Xdata, 'preprocessed headlines': preprocessed_Xdata})
            df = df.dropna(how='any',axis=0)
            df.to_csv(filepath, encoding='utf-8', index=False)
        data = pd.read_csv(filepath)
        data = data.dropna(how='any',axis=0)
        train_and_test_data(data)
    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)
    # print("No: of train positive samples: {}".format(len(train_data[train_data['is_sarcastic'] == 1])))
    # print("No: of train negative samples: {}".format(len(train_data[train_data['is_sarcastic'] == 0])))
    # print("No: of test positive samples: {}".format(len(test_data[test_data['is_sarcastic'] == 1])))
    # print("No: of test negative samples: {}".format(len(test_data[test_data['is_sarcastic'] == 0])))

    Xtrain, Ytrain = train_data['preprocessed headlines'], train_data['is_sarcastic']
    Xtest , Ytest  = test_data['preprocessed headlines'],  test_data['is_sarcastic']
    return Xtrain, Ytrain, Xtest , Ytest


def train_and_test_data(df):
    df = shuffle(df)
    cutoff = int(0.8*len(df))
    train_data = df.iloc[:cutoff]
    test_data = df.iloc[cutoff:]
    train_data.to_csv(train_path, encoding='utf-8', index=False)
    test_data.to_csv(test_path, encoding='utf-8', index=False)

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocessed_headline(headline):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    headline = headline.lower()
    remove_punc = tokenizer.tokenize(headline) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_headline = ' '.join(remove_stop)
    return updated_headline

def get_evaluation_data(test_data = True,length = 100):
    Xtrain,Ytrain,Xtest,Ytest = get_data()
    while True:
        idx = int(input())
        if test_data:
            if idx + length > len(Ytest):
                print("input an integer below: {}".format(len(Ytest)- length))
            else:
                headlines = Xtest[idx:idx+length]
                is_sarcastic = Ytest[idx:idx+length]
                return headlines, is_sarcastic
        else:
            if idx + length > len(Ytrain):
                print("input an integer below: {}".format(len(Ytrain)- length))
            else:
                headlines = Xtrain[idx:idx+length]
                is_sarcastic = Ytrain[idx:idx+length]
                return headlines, is_sarcastic