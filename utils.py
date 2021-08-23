import numpy as np

from hmmlearn.utils import log_mask_zero
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import special
from particles import kalman
from abc import ABC, abstractmethod 

  def kalman_simulate_data(F, Q, x_init, T, dim_states, G=None):
        true_data = np.zeros((1,T,dim_states))
        cur_x = x_init
        for k in range(T):
            # Check if a G transformation matrix is there
            if G is not None:
                v = np.matmul(G, np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q))
            else:
                v = np.random.multivariate_normal(np.zeros(dim_states), Q)
            true_data[0,k] = (F @ cur_x.reshape((dim_states, 1)) + v.reshape((dim_states,1))).reshape(dim_states,)
            cur_x = true_data[0,k]
        return true_data

    def kalman_make_data(H, R, true_data, T, dim_obs, dim_states):
        data = np.zeros((1,T,dim_obs))
        for k in range(true_data.shape[1]):
            data[0,k] = (np.matmul(H, true_data[0,k,:].reshape(dim_states,1)) + np.random.multivariate_normal(np.zeros(dim_obs), R).reshape((dim_obs,1))).reshape(dim_obs,)
        return data

    def dataset_to_sequences(X, T):
        # make sequence of length T
        # output dim : X.shape[0]*T

        result_X = []
        m,n=X.shape
        for i in range(T,m):
            result_X.append(X[i-T:i])
        result_X = np.array(result_X).reshape(len(result_X),T,n)
        return result_X

    def boost_gamma_randomforest(observations, gamma, n_states, threshold=0.80, fixed_threshold=True, percentage_threshold=False, debug=False):

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
