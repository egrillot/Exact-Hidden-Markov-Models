from RF-HMM import RF_HMM
import utils
import tensorflow as tf
import matplotlib.pyplot as plt

class LSTM():

    def __init__(self, input_dim, output_dim=1):
        # input_dim is a 2-tuple : (length of sequences, number of features)
        # if output_dim=1, the problem will be consider as a regression problem elif output_dim>1 it is consider as a classification model

        input=tf.keras.layers.Input(shape=(input_dim))
        x=tf.keras.layers.LSTM(128,return_sequences=True)(input)
        x=tf.keras.layers.LSTM(32)(x)
        x=tf.keras.layers.BatchNormalization()(x)
        if output_dim==1:
            x=tf.keras.layers.Dense(output_dim,activation='sigmoid')(X)
        else:
            x=tf.keras.layers.Dense(output_dim,activation='softmax')(x)
        self.model=tf.keras.Model(input,x)
    
    def train(self, X, Y, optimizer='adam', loss='mse', batch_size=32, epochs=100, validation_split=0.1, shuffle=True, save_path='', display=False):
        # train the LSTM model, save the model if needed by specifying a path 
        # X is a m*T*n array (T : length of the sequence, n : number of features)
    
        self.model.compile(optimizer=optimizer, loss=loss)
        if len(save_path)!=0:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            callbacks_list = [checkpoint]
            history=self.model.train(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=shuffle, verbose = 1,callbacks=callbacks_list)  
        else:
            history=self.model.train(X, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=shuffle, verbose = 1) 
        if display:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        
    
    def predict(self, X):
        # make prediction

        return self.encoder.predict(X)

class RF_HMM_LSTM(RF_HMM):

    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.LSTM=LSTM(input_dim, output_dim)
    
    def train(self, X, Y, T, split=0.7, display=False):
        # X is a m*n array ( n : number of features)
        # T is the length wished of sequence
        # Y is a m*output_dim array
        # split is the portion of data saved for the LSTM model training

        # split the training set for the rf-hmm training and the LSTM model training
        border=int((1-split)*X.shape[0])
        rf_hmm_train=X[:border]
        lstm_train=X[border:]
        y_lstm_train=Y[border:]

        # fit the rf-hmm model
        self.model.train(rf_hmm_train, display=display)
        lstm_train_prob, _, _,=self.model.eval(lstm_train)

        # fit the LSTM model
        lstm_train_prob_input=utils.dataset_to_sequences(lstm_train_prob, T)
        self.LSTM.train(lstm_train_prob_input, y_lstm_train, display=display)
    
    def predict(self, X, T):
        # make prediction 
        # X is a m*n array
        
        # prepare input sequences for the LSTM
        lstm_input, _, _,=utils.dataset_to_sequences(self.model.eval(X), T)

        # LSTM prediction
        return self.LSTM.predict(lstm_input)




        


