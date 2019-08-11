import numpy as np
import pandas as pd
from preprocess import Preprocess
from models import Models
import pickle
from sklearn.metrics import roc_auc_score

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 200

## Reading and Preprocessing Train Data
comment, target = Preprocess.readTrainData()
sequence, tokenizer, word2idx = Preprocess.tokenize(comment, MAX_VOCAB_SIZE)
padded = Preprocess.padSequences(sequence, MAX_SEQ_LENGTH)
# seq_sizes = pd.Series(np.array([len(each) for each in sequence]))
# print(seq_sizes.describe())


## Reading and Preprocessing Test Data
comment_test, target_test = Preprocess.readTestData()
sequence_test = tokenizer.texts_to_sequences(comment_test)
padded_test = Preprocess.padSequences(sequence_test, MAX_SEQ_LENGTH)
target_test = target_test.astype(bool)

## Creating Embedding Matrix
word2vec = Preprocess.getWord2Vec()
embedding_matrix = Preprocess.getEmbeddingMatrix(MAX_VOCAB_SIZE, word2idx, word2vec)
number_of_words = len(embedding_matrix)

#######################################################################################
# ## Fitting CNN Model
# cnn_model = Models.usingCNN(embedding_matrix, MAX_SEQ_LENGTH)
# cnn_model.fit(padded, target,
#             batch_size = 128,
#             validation_split = 0.2,
#             epochs = 1)
#
# cnn_model_accuracy = cnn_model.evaluate(padded_test, target_test)
# print('[INFO] Test Accuracy of CNN Model is {}'.format(round(cnn_model_accuracy[1], 2)))
#
# ## Calculating AUC score for CNN model
# auc_list = []
# predicted = cnn_model.predict(padded_test)
# for i in range(6):
#     auc = roc_auc_score(target_test[:, i], predicted[:, i])
#     auc_list.append(auc)
# mean_auc = np.mean(auc_list)
# print('[INFO] The mean AUC score over each tag is {}'.format(mean_auc))
#
# ## Saving the CNN Model
# cnn_model_json = cnn_model.to_json()
# with open('Models/cnn_model.json', 'w') as json_file:
#     json_file.write(cnn_model_json)
# cnn_model.save_weights('Models/cnn_model.h5')
#######################################################################################
# ## Fitting RNN Model
# rnn_model = Models.usingRNN(embedding_matrix, MAX_SEQ_LENGTH)
# rnn_model.fit(padded, target,
#             batch_size = 128,
#             validation_split = 0.2,
#             epochs = 1)
#
# rnn_model_accuracy = rnn_model.evaluate(padded_test, target_test)
# print('[INFO] Test Accuracy of RNN Model is {}'.format(round(rnn_model_accuracy[1], 2)))
#
# ## Calculating AUC score for RNN model
# auc_list = []
# predicted = rnn_model.predict(padded_test)
# for i in range(6):
#     auc = roc_auc_score(target_test[:, i], predicted[:, i])
#     auc_list.append(auc)
# mean_auc = np.mean(auc_list)
# print('[INFO] The mean AUC score over each tag is {}'.format(mean_auc))
#
# ## Saving the CNN Model
# rnn_model_json = rnn_model.to_json()
# with open('Models/rnn_model.json', 'w') as json_file:
#     json_file.write(rnn_model_json)
# rnn_model.save_weights('Models/rnn_model.h5')
#####################################################################################
## Fitting RNN Model
hybrid_model = Models.usingHybrid(embedding_matrix, MAX_SEQ_LENGTH)
hybrid_model.fit(padded, target,
            batch_size = 128,
            validation_split = 0.2,
            epochs = 5)

hybrid_model_accuracy = hybrid_model.evaluate(padded_test, target_test)
print('[INFO] Test Accuracy of Hybrid Model is {}'.format(round(hybrid_model_accuracy[1], 2)))

## Calculating AUC score for RNN model
auc_list = []
predicted = hybrid_model.predict(padded_test)
for i in range(6):
    auc = roc_auc_score(target_test[:, i], predicted[:, i])
    auc_list.append(auc)
mean_auc = np.mean(auc_list)
print('[INFO] The mean AUC score over each tag is {}'.format(mean_auc))

## Saving the CNN Model
hybrid_model_json = hybrid_model.to_json()
with open('Models/hybrid_model.json', 'w') as json_file:
    json_file.write(hybrid_model_json)
hybrid_model.save_weights('Models/hybrid_model.h5')
