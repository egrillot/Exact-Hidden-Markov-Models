from abc import ABC, abstractmethod 
import numpy as np
from KalmanFilter import KalmanFilterCustom
from particles import kalman
import utils

# Abstract models
class KalmanModel(ABC):

  # Transition model
  @abstractmethod
  def make_F(self):
    pass

  # Process noise
  def make_G(self):
    pass

  # Process noise
  @abstractmethod
  def make_Q(self):
    pass

  # Observation model
  @abstractmethod
  def make_H(self):
    pass

  # Observation noise
  @abstractmethod
  def make_R(self):
    pass

  # Get the linear gaussian parameters for the Kalman filter
  def get_kf_params(self, useG):
    # Get the linear gaussian model parameters
    F, Q, H, R = self.make_F(), self.make_Q(), self.make_H(), self.make_R()
    # If a intermediate G matrix is defined
    if useG:
      G = self.make_G()
      Q = np.matmul(G, np.matmul(Q, G.T))
    dim_states = F.shape[0]
    dim_obs = H.shape[0]
    return F, Q, H, R, np.zeros((1,dim_states,1)), np.eye(dim_states).reshape(1,dim_states,dim_states)

  # Make a kalman filter based on the model
  def make_kf(self, useG=False):
    # Get the linear gaussian model parameters
    F, Q, H, R, initial_value, initial_covariance = self.get_kf_params(useG)
    # Build the Kalman filter
    return KalmanFilterCustom(state_transition=F,
                              process_noise=Q,
                              observation_model=H,
                              observation_noise=R,
                              initial_value=initial_value,
                              initial_covariance=initial_covariance)

  # Simulate the gaussian linear model
  def simulate(self, T, particles=True, G=None):
    # Get the linear gaussian model parameters
    F, Q, H, R = self.make_F(), self.make_Q(), self.make_H(), self.make_R()
    dim_states = F.shape[0]
    dim_obs = H.shape[0]
    # Use particles
    if particles:
      # Build the state space model with particles
      ssm = kalman.MVLinearGauss(F=F, G=H, covX=Q, covY=R, mu0=np.zeros((dim_states,)), cov0=np.eye(dim_states))
      # Simulate the data
      true_data, data = ssm.simulate(T)
      # Reshape everything to match simdkalman
      data = np.array(data).reshape(1, T, dim_obs)
      true_data = np.array(true_data).reshape(1, T, dim_states)
    else:
      # Manualy do the work
      true_data = utils.kalman_simulate_data(F, Q, np.zeros((dim_states,1)), T, dim_states, G=G)
      data = utils.kalman_make_data(H, R, true_data, T, dim_obs, dim_states)
    return data, true_data, dim_states, dim_obs

# General model class
class GeneralModel(KalmanModel):

    def __init__(self, F, Q, H, R):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

    # Transition model
    def make_F(self):
        return self.F

    # Process noise
    def make_Q(self):
        return self.Q

    # Observation model
    def make_H(self):
        return self.H

    # Observation noise
    def make_R(self):
        return self.R

    # Make a kalman filter based on the model
    def make_kf(self):
        return super().make_kf()

    # Simulate the gaussian linear model
    def simulate(self, T):
        return super().simulate(T)

    def __str__(self):
        return "GeneralModel(F={},Q={},H={},R={})".format(self.F, self.Q, self.H, self.R)

# Dynamic model
class DynamicModel(KalmanModel):

    def __init__(self, Te, sigma_Q, sigma):
        self.Te = Te
        self.sigma_Q = sigma_Q
        self.sigma = sigma   

    # Transition model
    def make_F(self):
        ret = np.diag(np.ones(2))
        ret[0][1] = self.Te
        return ret

    # Process noise
    def make_Q(self):
        return (self.sigma_Q**2) * np.array([[(self.Te**3)/3, (self.Te**2)/2],
                                        [(self.Te**2)/2, self.Te]])

    # Observation model
    def make_H(self):
        return np.array([[1,0]])

    # Observation noise
    def make_R(self):
        return np.array([[self.sigma]])

    # Make a kalman filter based on the model
    def make_kf(self):
        return super().make_kf()

    # Simulate the gaussian linear model
    def simulate(self, T):
        return super().simulate(T)
    
    def __str__(self):
        return "DynamicModel(Te={},sigma_Q={},sigma={})".format(self.Te, self.sigma_Q, self.sigma)

# Standard Seasonal Adjustements model
class StandardSeasonalAdjustement(KalmanModel):

    def __init__(self, p, tau1, tau2, sigma):
        self.p = p
        self.tau1 = tau1
        self.tau2 = tau2
        self.sigma = sigma

    # Transition model
    def make_F(self):
        state_transition = np.zeros((self.p+1,self.p+1))
        state_transition[0,0] = 2
        state_transition[0,1] = 1
        state_transition[1,0] = 1
        state_transition[1,1] = -1
        state_transition[2,2:] = [-1.0] * (self.p-1)
        state_transition[3:,2:-1] = np.eye(self.p-2)
        return state_transition

    # Intermediate process noise matrix
    def make_G(self):
        ret = np.zeros((self.p+1, 2))
        ret[0,0] = 1
        ret[2,1] = 1
        return ret

    # Process noise
    def make_Q(self):
        return np.diag([self.tau1**2, self.tau2**2])

    # Observation model
    def make_H(self):
        ret = np.zeros((1, self.p+1))
        ret[0,0] = 1
        ret[0,2] = 1
        return ret

    # Observation noise
    def make_R(self):
        return np.array([[self.sigma**2]])

    # Make a kalman filter based on the model
    def make_kf(self):
        return super().make_kf(useG=True)

    # Simulate the gaussian linear model
    def simulate(self, T):
        return super().simulate(T, particles=False, G=self.make_G)
    
    def __str__(self):
        return "StandardSeasonalAdjustement(p={}, tau1={}, tau2={}, sigma={})".format(self.p, self.tau1, self.tau2, self.sigma)

