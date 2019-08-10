from keras.layers import Input, Embedding, Dense
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model

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
