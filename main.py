import numpy as np
import pandas as pd
from preprocess import Preprocess
from models import Models
import pickle
from sklearn.metrics import roc_auc_score
import sys

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 200

## Reading and Preprocessing Train Data
comment, target = Preprocess.readTrainData()
sequence, tokenizer, word2idx = Preprocess.tokenize(comment, MAX_VOCAB_SIZE)
padded = Preprocess.padSequences(sequence, MAX_SEQ_LENGTH)

## Reading and Preprocessing Test Data
comment_test, target_test = Preprocess.readTestData()
comment_test = [list(np.array([each for each in sentence.split(' ')]).ravel()) for sentence in comment_test]
sequence_test = tokenizer.texts_to_sequences(comment_test)
padded_test = Preprocess.padSequences(sequence_test, MAX_SEQ_LENGTH)
target_test = target_test.astype(bool)

## Creating Embedding Matrix
word2vec = Preprocess.getWord2Vec()
embedding_matrix = Preprocess.getEmbeddingMatrix(MAX_VOCAB_SIZE, word2idx, word2vec)
number_of_words = len(embedding_matrix)

## Saving the tokenizer
with open('Models/tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file, protocol = pickle.HIGHEST_PROTOCOL)
#########################################################################################
## Fitting Models
models = {'CNN' : Models.usingCNN(embedding_matrix, MAX_SEQ_LENGTH),
        'RNN' : Models.usingRNN(embedding_matrix, MAX_SEQ_LENGTH),
        'Hybrid' : Models.usingHybrid(embedding_matrix, MAX_SEQ_LENGTH),
        'MultiCNN' : Models.usingMCNN(embedding_matrix, MAX_SEQ_LENGTH),
        'MultiRNN' : Models.usingMRNN(embedding_matrix, MAX_SEQ_LENGTH)}

for each in models.keys():
    model = models[each]
    model.fit(padded, target,
                batch_size = 128,
                validation_split = 0.1,
                epochs = 5,
                shuffle = True)

    model_accuracy = model.evaluate(padded_test, target_test)
    print('[INFO] Test Accuracy of {} Model is {}'.format(each, round(model_accuracy[1], 2)))

    ## Calculating AUC score for models
    auc_list = []
    predicted = model.predict(padded_test)
    for i in range(6):
        auc = roc_auc_score(target_test[:, i], predicted[:, i])
        auc_list.append(auc)
    mean_auc = np.mean(auc_list)
    print('[INFO] The mean AUC score of {} Model over each tag is {}'.format(each, mean_auc))

    ## Saving the Models
    model_json = model.to_json()
    with open('Models/{}_model.json'.format(each.lower()), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('Models/{}_model.h5'.format(each.lower()))
