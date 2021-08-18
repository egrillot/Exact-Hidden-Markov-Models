from RF-HMM import RF_HMM
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import utils

class Encoder():

    def __init__(self, input_dim, type='low dimension'):
        # input_dim is a 2-tuple : (length of the sequence, number of features)

        self.type=type
        T,n=input_dim
        if type=='low dimension':
            model=tf.keras.models.Sequential()
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, activation='tanh', input_shape=(T,n))))
            model.add(tf.keras.layers.RepeatVector(T))
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, activation='tanh', return_sequences=True)))
            model.add(tf.keras.layers.Dense(n))
            self.model=model.build(input_shape=(None,T,n))
        if type=='high dimension':
            model=tf.keras.models.Sequential()
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='tanh', input_shape=(T,n))))
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, activation='tanh', return_sequences=True)))
            model.add(tf.keras.layers.RepeatVector(T))
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, activation='tanh', return_sequences=True)))
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)))
            model.add(tf.keras.layers.Dense(n))
            self.model=model.build(input_shape=(None,T,n))

    def train(self, X, optimizer='adam', loss='mse', batch_size=32, epochs=100, validation_split=0.1, shuffle=True, save_path='', display=False):
        # train the autoencoder, save the model if needed by specifying a path 
        # X is a m*T*n array (T : length of the sequence, n : number of features)

        self.model.compile(optimizer=optimizer, loss=loss)
        if len(save_path)!=0:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            callbacks_list = [checkpoint]
            history=self.model.train(X, X, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=shuffle, verbose = 1,callbacks=callbacks_list)  
        else:
            history=self.model.train(X, X, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=shuffle, verbose = 1) 
        if display:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        # build the encoder
                
        if self.type=='low dimension':
            self.encoder=tf.keras.Model(self.model.input,self.model.layers[0].output)
        if self.type=='high dimension':
            self.encoder=tf.keras.Model(self.model.input,self.model.layers[1].output)
    
    def predict(self, X):
        # make prediction

        return self.encoder.predict(X)

class Encoder_RF_HMM(RF_HMM):

    def __init__(self, input_dim, type='low simple'):
        super().__init__()
        self.encoder=Encoder(input_dim, type)

    def train(self, X, T, split=0.4, display=False):
        # X is a m*n array ( n : number of features)
        # T is the length wished of sequence
        # split is the portion of data saved for the rf_hmm training

        # split the training set for the encoder training and the rf-hmm training
        border=int((1-split)*X.shape[0])
        autoencoder_train=X[:border]
        rf_hmm_train=X[border:]

        # train the autoencoder
        autoencoder_train_input=utils.dataset_to_sequences(autoencoder_train, T)
        self.encoder.train(autoencoder_train_input, display)

        # train the rf-hmm
        rf_hmm_train_input=self.encoder.predict(rf_hmm_train)
        self.model.train(rf_hmm_train_input,display)
    
    def predict(self, X, T):
        # make prediction 

        # prepare data for the encoder
        autoencoder_input=utils.dataset_to_sequences(X, T)

        # encode the dataset
        rf_hmm_input=self.encoder.predict(autoencoder_input)

        # evaluation with the rf-hmm
        return self.model.eval(rf_hmm_input)
