from keras.layers import Input, Embedding, Dense, Concatenate
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional, Dropout, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import f1_score
import keras_metrics

class Models:
    def usingCNN(embedding_matrix,  max_seq_len):
        embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights = [embedding_matrix],
                            input_length = max_seq_len,
                            trainable = False)

        input = Input(shape = (max_seq_len,))
        x = embedding_layer(input)
        x = Conv1D(128, 3, activation = 'relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(128, 3, activation = 'relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(128, 3, activation = 'relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation = 'relu')(x)
        output = Dense(6, activation = 'sigmoid')(x)

        cnn_model = Model(input, output)
        cnn_model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return cnn_model

    def usingRNN(embedding_matrix,  max_seq_len):
        embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights = [embedding_matrix],
                            input_length = max_seq_len,
                            trainable = False)

        input = Input(shape = (max_seq_len,))
        x = embedding_layer(input)
        x = Bidirectional(LSTM(32, return_sequences = True))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation = 'relu')(x)
        x = Dropout(0.1)(x)
        output = Dense(6, activation = 'sigmoid')(x)

        rnn_model = Model(input, output)
        rnn_model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return rnn_model

    def usingHybrid(embedding_matrix,  max_seq_len):
        embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights = [embedding_matrix],
                            input_length = max_seq_len,
                            trainable = False)

        input = Input(shape = (max_seq_len,))
        x = embedding_layer(input)
        x = Conv1D(128, 3, activation = 'relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(128, 3, activation = 'relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(128, 3, activation = 'relu')(x)
        x = Bidirectional(LSTM(32, return_sequences = True))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation = 'relu')(x)
        output = Dense(6, activation = 'sigmoid')(x)

        hybrid_model = Model(input, output)
        hybrid_model.compile(optimizer = Adam(0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
        return hybrid_model

    def usingMCNN(embedding_matrix,  max_seq_len):
        embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights = [embedding_matrix],
                            input_length = max_seq_len,
                            trainable = False)

        input = Input(shape = (max_seq_len,))
        x = embedding_layer(input)

        x = SpatialDropout1D(0.2)(x)

        x1 = Conv1D(128, 1, activation = 'relu')(x)
        x2 = Conv1D(128, 2, activation = 'relu')(x)
        x3 = Conv1D(128, 3, activation = 'relu')(x)
        x4 = Conv1D(128, 4, activation = 'relu')(x)

        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(x4.shape)

        x1 = GlobalMaxPooling1D()(x1)
        x2 = GlobalMaxPooling1D()(x2)
        x3 = GlobalMaxPooling1D()(x3)
        x4 = GlobalMaxPooling1D()(x4)

        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(x4.shape)

        x = Concatenate(axis = 1)([x1, x2, x3, x4])
        x = Dropout(0.5)(x)
        x = Dense(128, activation = 'relu')(x)
        output = Dense(6, activation = 'sigmoid')(x)

        mcnn_model = Model(input, output)
        mcnn_model.compile(optimizer = Adam(lr=1e-3, decay=1e-6), loss = 'binary_crossentropy', metrics = ['accuracy'])
        return mcnn_model
