# RANDOM FOREST-HIDDEN MARKOV MODEL

This package provides an implementation of our algorithm **RF-HMM**. First, our algorithm will execute the Expectation-Maximization algorithm using hmmlearn and then we boost
the gamma probabilities to reach a better likelihood with the time series. If your time series is in high dimension (more than 4), we recommand to use the class **Encoder-RF-HMM**
where the data are compressed by an encoder before starting to train the RF-HMM. Indeed, the RF-HMM provides poor result in high dimension. And finally if you want to make 
prediction, you are invited to use the class **RF-HMM-LSTM** where an LSTM can be trained on your hidden states to make prediction.

## Usage

