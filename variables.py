seed = 42
vocab_size = 15000
max_length = 120
embedding_dim = 120
trunc_type = 'post'
pad_type = 'post'
oov_tok = "<OOV>"
num_epochs = 10
batch_size = 128
size_lstm1 = 128
size_lstm2 = 64
size_dense = 64
size_output = 1

#Data paths and weights
test_path = "test.csv"
train_path = "train.csv"
sentiment_path = "model.json"
sentiment_weights = "model.h5"
filepath = "train_test_data.csv"
data_path = "Sarcasm_Headlines_Dataset_v2.json"