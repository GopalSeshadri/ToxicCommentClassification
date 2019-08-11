import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Preprocess:
    stop_words = set(stopwords.words('english'))

    def preprocessText(sentence):
        sentence = ' '.join(sentence.split())
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(sentence)
        sentence = ' '.join([lemmatizer.lemmatize(word) for word in words if word not in Preprocess.stop_words])
        return sentence

    # Reading the Glove Vector
    def getWord2Vec():
        word2vec = {}
        with open('Embeddings/glove.6B.100d.txt', encoding = 'utf8') as file:
            for line in file:
                values = line.split()
                word2vec[values[0]] = values[1:]
        return word2vec

    # Reading Training Data
    def readTrainData():
        train_df = pd.read_csv('Data/train.csv')
        train_df = train_df.dropna()
        train_df['comment_text'] = train_df['comment_text'].apply(lambda x: Preprocess.preprocessText(x))
        comment = train_df['comment_text'].values
        target = train_df[['toxic',	'severe_toxic',	'obscene', 'threat', 'insult', 'identity_hate']].values
        return comment, target

    # Reading Training Data
    def readTestData():
        test_df = pd.read_csv('Data/test.csv')
        test_df = test_df.dropna()
        test_df['comment_text'] = test_df['comment_text'].apply(lambda x: Preprocess.preprocessText(x))
        comment = test_df['comment_text'].values

        test_labels_df = pd.read_csv('Data/test_labels.csv')
        target = test_labels_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        return comment, target

    # Tokenize the data
    def tokenize(data, vocab_size):
        tokenizer = Tokenizer(num_words = vocab_size)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        return sequences, tokenizer, tokenizer.word_index

    def padSequences(sequence, max_seq_length):
        return pad_sequences(sequence, maxlen = max_seq_length)

    # Get the embedding matrix
    def getEmbeddingMatrix(max_vocab, word2idx, word2vec):
        number_of_words = min(max_vocab, len(word2idx) + 1)
        embedding_matrix = np.zeros((number_of_words, 100)) # Here 100 is the dimension of GloVe Embeddings
        for word, idx in word2idx.items():
            if idx < max_vocab:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
        return embedding_matrix
