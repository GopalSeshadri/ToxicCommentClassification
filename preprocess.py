import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Preprocess:
    stop_words = set(stopwords.words('english'))

    def preprocessText(sentence):
        '''
        This function takes in a sentence and it returns a processed sentence

        Parameters:
        sentence (str) : Text to process

        Returns:
        sentence (str) : Processed Text
        '''
        sentence = sentence.lower()
        #remove \n
        sentence = re.sub('\\n','',sentence)
        # remove leaky elements like ip,user
        sentence = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}','',sentence)
        #removing usernames
        sentence = re.sub('\[\[.*\]','',sentence)
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(sentence)
        sentence = ' '.join([lemmatizer.lemmatize(word) for word in words if word not in Preprocess.stop_words])
        return sentence

    # Reading the Glove Vector
    def getWord2Vec():
        '''
        This function returns the dictionary of word vectors

        Returns:
        word2vec (dict) : A dictionary of word vectors
        '''
        word2vec = {}
        with open('Embeddings/glove.6B.100d.txt', encoding = 'utf8') as file:
            for line in file:
                values = line.split()
                word2vec[values[0]] = values[1:]
        return word2vec

    # Reading Training Data
    def readTrainData():
        '''
        This function returns the training data predictors and response variables

        Returns:
        comment (list) : A list that contains the input sentences. This is the predictor variable
        target (list) : This is a list that contains the target labels.
        '''
        train_df = pd.read_csv('Data/train.csv')
        train_df = train_df.dropna()
        train_df['comment_text'] = train_df['comment_text'].apply(lambda x: Preprocess.preprocessText(x))
        train_df['sum'] = train_df['toxic'] + train_df['severe_toxic'] + train_df['obscene'] + train_df['threat'] + train_df['insult'] + train_df['identity_hate']
        train_df['good'] = train_df['sum'].apply(lambda x: 1 if x == 0 else 0)
        comment = train_df['comment_text'].values
        target = train_df[['toxic',	'severe_toxic',	'obscene', 'threat', 'insult', 'identity_hate']].values
        return comment, target

    # Reading Testing Data
    def readTestData():
        '''
        This function returns the testing data predictors and response variables

        Returns:
        comment (list) : A list that contains the input sentences. This is the predictor variable
        target (list) : This is a list that contains the target labels.
        '''
        test_df = pd.read_csv('Data/test.csv')
        test_df = test_df.dropna()
        test_df['comment_text'] = test_df['comment_text'].apply(lambda x: Preprocess.preprocessText(x))
        comment = test_df['comment_text'].values

        test_labels_df = pd.read_csv('Data/test_labels.csv')
        test_labels_df['sum'] = test_labels_df['toxic'] + test_labels_df['severe_toxic'] + test_labels_df['obscene'] + test_labels_df['threat'] + test_labels_df['insult'] + test_labels_df['identity_hate']
        test_labels_df['good'] = test_labels_df['sum'].apply(lambda x: 1 if x == 0 else 0)
        target = test_labels_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        return comment, target

    # Tokenize the data
    def tokenize(data, vocab_size):
        '''
        This function takes in data and vocabulary size and returns the tokenized sequence, tokenizer object and word indices
        of the tokenizer.

        Parameters:
        data (list) : A list that contains the input sentences.
        vocab_size (int) : The size of vocabulary.

        Returns:
        sequences (list) : A list of tokenized data records.
        tokenizer (object) : A tokenizer object
        word_index (dict) : A dictionary of tokenized data
        '''
        tokenizer = Tokenizer(num_words = vocab_size)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        return sequences, tokenizer, tokenizer.word_index

    def padSequences(sequence, max_seq_length):
        '''
        This function takes in data sequence and maximum sequence length. It returns the padded sequences with maximum length.

        Parameters:
        sequences (list) : A list of tokenized data records.
        max_seq_length (int) : The maximum length of the input sequence

        Returns:
        padded_sequences (list) : A list of padded tokenized data
        '''
        return pad_sequences(sequence, maxlen = max_seq_length)

    # Get the embedding matrix
    def getEmbeddingMatrix(max_vocab, word2idx, word2vec):
        '''
        This function takes in maximum vocabulary size, word2idx and word2vec and it returns the embedding matrix.

        Parameters:
        max_vocab (int) : Maximum vocabulary size.
        word2idx (dict) : A dictionary of tokenized data
        word2vec (dict) : A dictionay of word vectors

        Returns:
        embedding_matrix (numpy array) : A matrix of embedding vectors
        '''
        number_of_words = min(max_vocab, len(word2idx) + 1)
        embedding_matrix = np.zeros((number_of_words, 100)) # Here 100 is the dimension of GloVe Embeddings
        embedding_matrix[0] = np.zeros((1, embedding_matrix.shape[1]))
        for word, idx in word2idx.items():
            if idx < max_vocab:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
        return embedding_matrix

    ## Calculating Naive Bayes word count ratio
    def pr(dtm, y, y_i):
        '''
        This function takes in document term matrix, target y values and y class labels.
        It returns the count ratio.
        '''
        p = dtm[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)
