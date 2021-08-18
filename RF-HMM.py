import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import _BaseHMM
from hmmlearn.utils import log_mask_zero
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from scipy import special

class utils():

    def boost_gamma_randomforest(observations, gamma, n_states, threshold=0.80, fixed_threshold=True, 
                                 percentage_threshold=False, debug=False):

        # Compute max and argmax
        temp = np.array([np.max(i) for i in gamma])
        y = np.array([np.argmax(i) for i in gamma])

        # Get a good threshold
        if not fixed_threshold and not percentage_threshold:
            threshold = 1
            for n in range(n_states):
                threshold = min(np.max(temp[y == n]), threshold)

        # Train only on the best
        if not fixed_threshold and percentage_threshold:
            temp_s = np.sort(temp)
            threshold = temp_s[int(0.75 * temp.shape[0])]

        if debug:
            print("XGB Threshold at {}".format(threshold))

        # Make dataset
        y = y[temp >= threshold]
        X = observations[temp >= threshold]

        # Fit model
        if debug:
            print("XGB Training on {} observations".format(X.shape[0]))
        model = RandomForestClassifier(n_estimators=300)
        model.fit(X, y)

        # Make prediction
        prediction = model.predict_proba(observations)

        # Compute accuracy
        if debug:
            print("XGB accuracy : {}".format(accuracy_score(np.array([np.argmax(i) for i in gamma]), np.array([np.argmax(i) for i in prediction]))))

        # Make prediction
        return prediction

    def real_predict_proba(model, X):
    # Real predict proba
    # Do the forward pass manualy

        # Compute the log-likelihood
        framelogprob = model._compute_log_likelihood(X)
        # Do the forward pass
        _, fwdlattice = model._do_forward_pass(framelogprob)
        # Get the log-probabilities
        return fwdlattice

    def predict_model(model, X_test):
        # Get the hidden states
        hidden_states_proba = utils.real_predict_proba(model, X_test) # model.predict_proba(X_test)
        # Get log transition matrix
        log_T = log_mask_zero(model.transmat_)
        # Get the predicted stuff
        predicted_means = np.zeros((X_test.shape[0]+1, X_test.shape[1]))
        predicted_state = np.zeros((X_test.shape[0]+1, 1), dtype=int)
        predicted_proba = np.zeros((X_test.shape[0]+1, hidden_states_proba.shape[1]))
        for i in range(1,X_test.shape[0]+1):
            # Compute predicted log probability
            work_buffer = np.zeros((hidden_states_proba.shape[1],hidden_states_proba.shape[1]))
            for k in range(hidden_states_proba.shape[1]):
                for l in range(hidden_states_proba.shape[1]):
                    work_buffer[k,l] = hidden_states_proba[i-1,l] + log_T[k,l]
                predicted_proba[i,k] = special.logsumexp(work_buffer[k,:])
            # Get the corresponding state
            predicted_state[i] = int(np.argmax(predicted_proba[i]))
            # Get the corresponding observation mean
            predicted_means[i] = model.means_[predicted_state[i]]
        # Return everything
        return predicted_proba[:-1], predicted_state[:-1], predicted_means[:-1]
    
    def eval_model(model, X_test):
        # Make the prediction
        return utils.predict_model(model, X_test)

class CustomGaussianHmm(GaussianHMM):

    def __init__(self, threshold, fixed_threshold, percentage_threshold):
        super().__init__()
        self.threshold=threshold
        self.fixed_threshold=fixed_threshold
        self.percentage_threshold=percentage_threshold

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
      
        # Call BaseHMM
        '''
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = np.full((n_components, n_components), -np.inf)
            xi_ij = compute_log_xi_custom(n_samples, n_components, fwdlattice,
                                      log_mask_zero(self.transmat_),
                                      bwdlattice, framelogprob,
                                      log_xi_sum)
            
            # Do the magic
            #log_xi_ij = np.log(boost_lstm(obs, np.exp(xi_ij), n_components))
            #log_xi_ij = boost_ffnn(obs, xi_ij, n_components)
            log_xi_ij = np.array(boost_gamma_randomforest_multioutput(obs, np.exp(xi_ij), n_components))
            print(log_xi_ij.T.shape)

            # Do the sum
            compute_log_xi_sum_custom(log_xi_ij, n_samples, n_components, fwdlattice, log_xi_sum)

            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)
        '''
        _BaseHMM._accumulate_sufficient_statistics(self,
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        
        # XGB
        posteriors = utils.boost_gamma_randomforest(obs, posteriors, fwdlattice.shape[1], self.threshold, self.fixed_threshold, self.percentage_threshold)
        
        # Classic Gaussian HMM
        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

class RF_HMM():
    
    def __init__(self, n_states,covars_type='diag', evaluation_type='acc', params='stmc', threshold=0.70, fixed_threshold=True, percentage_threshold=False):
        self.model=CustomGaussianHmm(n_components=n_states, covariance_type=covars_type, params=params,
                                     threshold=threshold, fixed_threshold=fixed_threshold, percentage_threshold=percentage_threshold)
        if evaluation_type=='acc':
            self.evaluation_type=accuracy_score
        if evaluation_type=='mae':
            self.evaluation_type=mean_absolute_error
        if evaluation_type=='mse':
            self.evaluation_type=mean_squared_error

    def train(self, time_series, display=False):
        # time_series is a n*m array
        # train self.model
        
        self.model.fit(time_series)
        if display:
            plt.figure(figsize=(20,15))
            plt.plot(self.model.monitor_.history, label="Custom model")
            plt.title("log-likelihood during EM")
            plt.xlabel("Iterations")
            plt.ylabel("log-likelihood")
            plt.legend()
            plt.show()
    
    def eval(self, time_series):
        # time_series is a n*m array
        # evaluate the model on the time_series array

        return utils.eval_model(self.model, time_series)
 

    
