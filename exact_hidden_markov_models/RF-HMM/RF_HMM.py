import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import _BaseHMM
from ..utils import *


class CustomGaussianHmm(GaussianHMM):

    def __init__(self, n_components, covariance_type, params, threshold, fixed_threshold, percentage_threshold):
        super().__init__(n_components=n_components,covariance_type=covariance_type,params=params)
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
        posteriors = boost_gamma_randomforest(obs, posteriors, fwdlattice.shape[1], threshold=0.70, fixed_threshold=True, percentage_threshold=False)
        
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
    
    def __init__(self,n_states,covars_type='diag',params='stmc',threshold=0.70,fixed_threshold=True,percentage_threshold=False):
        self.model = CustomGaussianHmm(n_components=n_states,covariance_type=covars_type,params=params,threshold=threshold,fixed_threshold=fixed_threshold,percentage_threshold=percentage_threshold)

    def train(self,time_series,display=False):
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
    
    def eval(self,time_series):
        # time_series is a n*m array
        # evaluate the model on the time_series array

        return eval_model(self.model,time_series)


    
