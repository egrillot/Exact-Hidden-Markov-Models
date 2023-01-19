from ..utils import dataset_to_sequences

from keras.layers import Embedding, LSTM, BatchNormalization, Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from .RF_HMM import RF_HMM

import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

class CustomLSTM():

    def __init__(self, input_dim, embedding_dim=0, output_embedding=0, output_dim=1) -> None:
        # input_dim is a 2-tuple : (length of sequences, number of features)
        # if embedding_dim>0 then the value is consider as the size of the set of values can be taken by your time series and a embedding layer is added
        # and the value T must specified as the length of your sequences 
        #  /!\ If you aren't working with an integer time series, don't use it /!\
        # if output_dim=1, the problem will be consider as a regression problem elif output_dim>1 it is consider as a classification model

        T, _ = input_dim
        input = Input(shape=input_dim)
        if embedding_dim!=0:
            x = Embedding(input_dim=embedding_dim,output_dim=output_embedding, input_length=T)(input)
            x = LSTM(128,return_sequences=True)(x)
            x = LSTM(32)(x)
        else:
            x = LSTM(128,return_sequences=True)(input)
            x = LSTM(32)(x)
            x = BatchNormalization()(x)
        if output_dim==1:
            x = Dense(output_dim,activation='sigmoid')(x)
        else:
            x = Dense(output_dim,activation='softmax')(x)
        self.model = Model(input,x)
    
    def train(self, X, Y, optimizer='adam', loss='mse', batch_size=32, epochs=100, validation_split=0.1, shuffle=True, save_path='', display=False) -> None:
        # train the LSTM model, save the model if needed by specifying a path 
        # X is a m*T*n array (T : length of the sequence, n : number of features)
    
        self.model.compile(optimizer=optimizer, loss=loss)
        if len(save_path)!=0:
            checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            callbacks_list = [checkpoint]
            history=self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=shuffle, verbose = 1,callbacks=callbacks_list)  
        else:
            history=self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=shuffle, verbose = 1) 
        if display:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        
    
    def predict(self, X) -> np.ndarray:
        # make prediction

        return self.model.predict(X)

class RF_HMM_LSTM:

    def __init__(self, n_states, T, embedding_dim=0, output_embedding=0, output_dim=1) -> None:
        self.rf_hmm = RF_HMM(n_states=n_states)
        input_dim=(T, n_states)
        self.LSTM = CustomLSTM(input_dim, embedding_dim, output_embedding, output_dim)
    
    def train(self, X, Y, T, split=0.7, display=False, epochs=100) -> None:
        # X is a m*n array ( n : number of features)
        # T is the length wished of sequence
        # Y is a m * output_dim array
        # split is the portion of data saved for the LSTM model training

        # split the training set for the rf-hmm training and the LSTM model training
        border = int((1-split) * X.shape[0])
        rf_hmm_train = X[:border]
        lstm_train = X[border:]
        y_lstm_train = Y[border:]

        # fit the rf-hmm model
        self.rf_hmm.train(rf_hmm_train, display=display)
        lstm_train_prob, _, _, = self.rf_hmm.eval(lstm_train)

        # fit the LSTM model
        lstm_train_prob_input = dataset_to_sequences(lstm_train_prob, T)
        self.LSTM.train(lstm_train_prob_input, y_lstm_train, display=display, epochs=epochs)
    
    def predict(self, X, T) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # make prediction 
        # X is a m*n array
        
        # prepare input sequences for the LSTM
        lstm_input = dataset_to_sequences(self.rf_hmm.eval(X)[0], T)

        # LSTM prediction
        return self.LSTM.predict(lstm_input)




        


