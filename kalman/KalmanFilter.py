import numpy as np
import matplotlib.pyplot as plt
import simdkalman

from particles import kalman
from simdkalman.primitives import ensure_matrix, douter, ddot_t_right, ddot, dinv
from simdkalman.kalmanfilter import em_initial_state

class KalmanFilterCustom(simdkalman.KalmanFilter):
    def __init__(self,
        state_transition,
        process_noise,
        observation_model,
        observation_noise,
        initial_value,
        initial_covariance):

        # Invoke the super of the simdkalman class
        # super().__init__(state_transition, process_noise, observation_model, observation_noise)
  
        state_transition = ensure_matrix(state_transition)
        n_states = state_transition.shape[-2]

        process_noise = ensure_matrix(process_noise, n_states)
        observation_model = ensure_matrix(observation_model)
        n_obs = observation_model.shape[-2]
        observation_noise = ensure_matrix(observation_noise, n_obs)

        assert(state_transition.shape[-2:] == (n_states, n_states))
        assert(process_noise.shape[-2:] == (n_states, n_states))
        assert(observation_model.shape[-2:] == (n_obs, n_states))
        assert(observation_noise.shape[-2:] == (n_obs, n_obs))

        self.state_transition = state_transition
        self.process_noise = process_noise
        self.observation_model = observation_model
        self.observation_noise = observation_noise

        # Assert that the initial values and covariance are well made
        # stolen from self.compute
        self.initial_value = initial_value
        self.initial_covariance = initial_covariance
        if self.initial_value is None:
            self.initial_value = np.zeros((n_states, 1))
        self.initial_value = ensure_matrix(initial_value)
        if len(self.initial_value.shape) == 1:
            self.initial_value = self.initial_value.reshape((n_states, 1))

        if self.initial_covariance is None:
            self.initial_covariance = ensure_matrix(
                np.trace(ensure_matrix(self.observation_model))*(5**2), n_states)

        self.initial_covariance = ensure_matrix(self.initial_covariance, n_states)
        self.initial_value = ensure_matrix(self.initial_value)
        assert(self.initial_value.shape[-2:] == (n_states, 1))
        assert(self.initial_covariance.shape[-2:] == (n_states, n_states))

        #if len(self.initial_value.shape) == 2:
        #   self.initial_value = np.vstack([self.initial_value[np.newaxis,...]]*n_vars)

        #if len(self.initial_covariance.shape) == 2:
        #    self.initial_covariance = np.vstack([self.initial_covariance[np.newaxis,...]]*n_vars)

    def predict(self,
        data,
        n_test,
        states = True,
        observations = True,
        covariances = True,
        verbose = False):
        return self.compute(
            data,
            n_test,
            smoothed = False,
            states = states,
            covariances = covariances,
            observations = observations,
            verbose = verbose).predicted

    def smooth(self,
        data,
        observations = True,
        states = True,
        covariances = True,
        verbose = False):
        return self.compute(
            data,
            0,
            smoothed = True,
            states = states,
            covariances = covariances,
            observations = observations,
            verbose = verbose).smoothed
    
    def compute(self,
        data,
        n_test,
        smoothed = True,
        filtered = False,
        states = True,
        covariances = True,
        observations = True,
        likelihoods = False,
        gains = False,
        log_likelihood = False,
        verbose = False):

        # pylint: disable=W0201
        result = simdkalman.KalmanFilter.Result()

        data = ensure_matrix(data)
        single_sequence = len(data.shape) == 1
        if single_sequence:
            data = data[np.newaxis,:]

        n_vars = data.shape[0]
        n_measurements = data.shape[1]
        n_states = self.state_transition.shape[-2]
        n_obs = self.observation_model.shape[-2]

        def empty_gaussian(
            n_states=n_states,
            n_measurements=n_measurements,
            cov=covariances):
            return simdkalman.kalmanfilter.Gaussian.empty(n_states, n_vars, n_measurements, cov)

        def auto_flat_observations(obs_gaussian):
            r = obs_gaussian
            if n_obs == 1:
                r = r.unvectorize_state()
            if single_sequence:
                r = r.unvectorize_vars()
            return r

        def auto_flat_states(obs_gaussian):
            if single_sequence:
                return obs_gaussian.unvectorize_vars()
            return obs_gaussian

        m = self.initial_value
        P = self.initial_covariance

        keep_filtered = filtered or smoothed
        if filtered or gains:
            result.filtered = simdkalman.KalmanFilter.Result()

        if log_likelihood:
            result.log_likelihood = np.zeros((n_vars,))
            if likelihoods:
                result.log_likelihoods = np.empty((n_vars, n_measurements))

        if keep_filtered:
            if observations:
                filtered_observations = empty_gaussian(n_states=n_obs)
            filtered_states = empty_gaussian(cov=True)

        if gains:
            result.filtered.gains = np.empty((n_vars, n_measurements, n_states, n_obs))

        for j in range(n_measurements):
            if verbose:
                print('filtering %d/%d' % (j+1, n_measurements))

            y = data[:,j,...].reshape((n_vars, n_obs, 1))

            tup = self.update(m, P, y, log_likelihood)
            m, P, K = tup[:3]
            if log_likelihood:
                l = tup[-1]
                result.log_likelihood += l
                if likelihoods:
                    result.log_likelihoods[:,j] = l

            if keep_filtered:
                if observations:
                    obs_mean, obs_cov = self.predict_observation(m, P)
                    filtered_observations.mean[:,j,:] = obs_mean[...,0]
                    if covariances:
                        filtered_observations.cov[:,j,:,:] = obs_cov

                filtered_states.mean[:,j,:] = m[...,0]
                filtered_states.cov[:,j,:,:] = P

            if gains:
                result.filtered.gains[:,j,:,:] = K

            m, P = self.predict_next(m, P)

        if smoothed:
            result.smoothed = simdkalman.KalmanFilter.Result()
            if states:
                result.smoothed.states = empty_gaussian()

                # lazy trick to keep last filtered = last smoothed
                result.smoothed.states.mean = 1*filtered_states.mean
                if covariances:
                    result.smoothed.states.cov = 1*filtered_states.cov

            if observations:
                result.smoothed.observations = empty_gaussian(n_states=n_obs)
                result.smoothed.observations.mean = 1*filtered_observations.mean
                if covariances:
                    result.smoothed.observations.cov = 1*filtered_observations.cov

            if gains:
                result.smoothed.gains = np.zeros((n_vars, n_measurements, n_states, n_states))
                result.pairwise_covariances = np.zeros((n_vars, n_measurements, n_states, n_states))

            ms = filtered_states.mean[:,-1,:][...,np.newaxis]
            Ps = filtered_states.cov[:,-1,:,:]

            for j in range(n_measurements)[-2::-1]:
                if verbose:
                    print('smoothing %d/%d' % (j+1, n_measurements))
                m0 = filtered_states.mean[:,j,:][...,np.newaxis]
                P0 = filtered_states.cov[:,j,:,:]

                PsNext = Ps
                ms, Ps, Cs = self.smooth_current(m0, P0, ms, Ps)

                if states:
                    result.smoothed.states.mean[:,j,:] = ms[...,0]
                    if covariances:
                        result.smoothed.states.cov[:,j,:,:] = Ps

                if observations:
                    obs_mean, obs_cov = self.predict_observation(ms, Ps)
                    result.smoothed.observations.mean[:,j,:] = obs_mean[...,0]
                    if covariances:
                        result.smoothed.observations.cov[:,j,:,:] = obs_cov

                if gains:
                    result.smoothed.gains[:,j,:,:] = Cs
                    result.pairwise_covariances[:,j,:,:] = ddot_t_right(PsNext, Cs)

        if filtered:
            if states:
                result.filtered.states = simdkalman.kalmanfilter.Gaussian(filtered_states.mean, None)
                if covariances:
                    result.filtered.states.cov = filtered_states.cov
                result.filtered.states = auto_flat_states(result.filtered.states)
            if observations:
                result.filtered.observations = auto_flat_observations(
                    filtered_observations)

        if smoothed:
            if observations:
                result.smoothed.observations = auto_flat_observations(
                    result.smoothed.observations)
            if states:
                result.smoothed.states = auto_flat_states(
                    result.smoothed.states)

        if n_test > 0:
            result.predicted = simdkalman.KalmanFilter.Result()
            if observations:
                result.predicted.observations = empty_gaussian(
                    n_measurements=n_test,
                    n_states=n_obs)
            if states:
                result.predicted.states = empty_gaussian(n_measurements=n_test)

            for j in range(n_test):
                if verbose:
                    print('predicting %d/%d' % (j+1, n_test))
                if states:
                    result.predicted.states.mean[:,j,:] = m[...,0]
                    if covariances:
                        result.predicted.states.cov[:,j,:,:] = P
                if observations:
                    obs_mean, obs_cov = self.predict_observation(m, P)
                    result.predicted.observations.mean[:,j,:] = obs_mean[...,0]
                    if covariances:
                        result.predicted.observations.cov[:,j,:,:] = obs_cov

                m, P = self.predict_next(m, P)

            if observations:
                result.predicted.observations = auto_flat_observations(
                    result.predicted.observations)
            if states:
                result.predicted.states = auto_flat_states(result.predicted.states)

        return result

    def em_state_transition(self, result, verbose=False):
        n_vars, n_measurements, n_states = result.smoothed.states.mean.shape

        C = np.zeros((n_vars, n_states, n_states))
        phi = np.zeros((n_vars, n_states, n_states))

        for j in range(n_measurements):
            if verbose:
                print('computing ML state transition, step %d/%d' % (j+1, n_measurements))

            ms1 = result.smoothed.states.mean[:,j,:][...,np.newaxis]
            Ps1 = result.smoothed.states.cov[:,j,...]

            if j > 0:
                phi += Ps1 + douter(ms1, ms1)
                V1 = result.pairwise_covariances[:,j,...]
                C += V1 + ddot_t_right(ms1, ms0)

            ms0 = ms1
            Ps0 = Ps1

        C *= (1.0 / (n_measurements - 1))
        phi *= (1.0 / (n_measurements - 1))

        return ddot(C, dinv(phi))

    def em_observation_model(self, result, data, verbose=False):
        n_vars, n_measurements, _ = result.smoothed.states.mean.shape
        n_obs = self.observation_model.shape[-2]
        n_states = self.state_transition.shape[-2]

        B = np.zeros((n_vars,n_obs,n_states))
        sigma = np.zeros((n_vars,n_states,n_states))
        n_not_nan = np.zeros((n_vars,))

        for j in range(n_measurements):
            if verbose:
                print('computing ML observation model, step %d/%d' % (j+1, n_measurements))

            ms = result.smoothed.states.mean[:,j,:][...,np.newaxis]
            Ps = result.smoothed.states.cov[:,j,...]

            y = data[:,j,...].reshape((n_vars, n_obs, 1))
            not_nan = np.ravel(np.all(~np.isnan(y), axis=1))
            n_not_nan += not_nan
            
            r_sigma = Ps + douter(ms, ms)
            sigma[not_nan,...] += r_sigma[not_nan,...]

            r_B = ddot_t_right(y, ms)
            B[not_nan,...] += r_B[not_nan,...]

        B /= np.maximum(n_not_nan, 1)[:, np.newaxis, np.newaxis]
        sigma /= np.maximum(n_not_nan, 1)[:, np.newaxis, np.newaxis]

        B = B.reshape((n_vars,n_obs,n_states))
        sigma = sigma.reshape((n_vars,n_states,n_states))

        return ddot(B, dinv(sigma))

    def em(self,
          data,
           n_iter=5,
           verbose=False,
           likelihoods=None,
           estimate_state_transition=True,
           estimate_process_noise=True,
           estimate_observation_model=True,
           estimate_observation_noise=True,
           estimate_initials=True,
           callbacks=None):
      
      # Ensure data is fine
      data = ensure_matrix(data)
      if len(data.shape) == 1:
        data = data[np.newaxis,:]

      # Global variables
      n_vars = data.shape[0]

      # Make a global mask to only perform EM on target time series
      vars_mask = np.ones((n_vars,), dtype=bool)
      vars_idx = np.arange(n_vars)

      # Get the matrices currently found
      cur_state_transition = np.copy(self.state_transition)
      cur_process_noise = np.copy(self.process_noise)
      cur_observation_model = np.copy(self.observation_model)
      cur_observation_noise = np.copy(self.observation_noise)
      cur_initial_value = np.copy(self.initial_value)
      cur_initial_covariance = np.copy(self.initial_covariance)

      # Process all the callbacks
      if verbose:
        print("Initialization")
      if callbacks is not None:
        for callback in callbacks:
          vars_idx = callback(self, vars_idx)

      new_model = KalmanFilterCustom(
          cur_state_transition[vars_idx],
          cur_process_noise[vars_idx],
          cur_observation_model[vars_idx],
          cur_observation_noise[vars_idx],
          cur_initial_value[vars_idx],
          cur_initial_covariance[vars_idx])
        
      return new_model.em_(data,
           vars_idx,
           cur_state_transition,
           cur_process_noise,
           cur_observation_model,
           cur_observation_noise,
           cur_initial_value,
           cur_initial_covariance,
           n_iter=n_iter,
           verbose=verbose,
           likelihoods=likelihoods,
           estimate_state_transition=estimate_state_transition,
           estimate_process_noise=estimate_process_noise,
           estimate_observation_model=estimate_observation_model,
           estimate_observation_noise=estimate_observation_noise,
           estimate_initials=estimate_initials,
           callbacks=callbacks)
  
    def em_(self,
           data,
           vars_idx,
           cur_state_transition,
           cur_process_noise,
           cur_observation_model,
           cur_observation_noise,
           cur_initial_value,
           cur_initial_covariance,
           n_iter=5,
           verbose=False,
           likelihoods=None,
           estimate_state_transition=True,
           estimate_process_noise=True,
           estimate_observation_model=True,
           estimate_observation_noise=True,
           estimate_initials=True,
           callbacks=None):

        if verbose:
          print("Training on {} timeseries".format(vars_idx.shape[0]))
        if (n_iter <= 0) or (vars_idx.shape[0] == 0):
            return KalmanFilterCustom(
                        cur_state_transition,
                        cur_process_noise,
                        cur_observation_model,
                        cur_observation_noise,
                        cur_initial_value,
                        cur_initial_covariance)

        if verbose:
            print("--- EM algorithm %d iteration(s) to go" % n_iter)
            print(" * E step")

        e_step = self.compute(
            data[vars_idx],
            n_test = 0,
            smoothed = True,
            filtered = False,
            states = True,
            observations = True,
            covariances = True,
            likelihoods = True,
            gains = True,
            log_likelihood = True,
            verbose = verbose)

        if isinstance(likelihoods, list):
          likelihoods.append(e_step.log_likelihood[0])

        if verbose:
            print(" * M step")

        if estimate_state_transition:
          cur_state_transition[vars_idx] = self.em_state_transition(e_step, verbose=verbose)[0]
        else:
          cur_state_transition[vars_idx] = self.state_transition
        
        if estimate_process_noise:
          cur_process_noise[vars_idx] = self.em_process_noise(e_step, verbose=verbose)
        else:
          cur_process_noise[vars_idx] = self.process_noise
        
        if estimate_observation_model:
          cur_observation_model[vars_idx] = self.em_observation_model(e_step, data[vars_idx], verbose=verbose)[0]
        else:
          cur_observation_model[vars_idx] = self.observation_model

        if estimate_observation_noise:
          cur_observation_noise[vars_idx] = self.em_observation_noise(e_step, data[vars_idx], verbose=verbose)
        else:
          cur_observation_noise[vars_idx] = self.observation_noise

        if estimate_initials:
          cur_initial_value[vars_idx], cur_initial_covariance[vars_idx] = em_initial_state(e_step, self.initial_value)
        else:
          cur_initial_value[vars_idx], cur_initial_covariance[vars_idx] = self.initial_value, self.initial_covariance

        # Process all the callbacks
        if callbacks is not None:
          for callback in callbacks:
            vars_idx = callback(self, vars_idx)

        new_model = KalmanFilterCustom(
            cur_state_transition[vars_idx],
            cur_process_noise[vars_idx],
            cur_observation_model[vars_idx],
            cur_observation_noise[vars_idx],
            cur_initial_value[vars_idx],
            cur_initial_covariance[vars_idx])

        return new_model.em_(data,
                                vars_idx,
                                cur_state_transition,
                                cur_process_noise,
                                cur_observation_model,
                                cur_observation_noise,
                                cur_initial_value,
                                cur_initial_covariance,
                                n_iter=n_iter-1,
                                verbose=verbose,
                                likelihoods=likelihoods,
                                estimate_state_transition=estimate_state_transition,
                                estimate_process_noise=estimate_process_noise,
                                estimate_observation_model=estimate_observation_model,
                                estimate_observation_noise=estimate_observation_noise,
                                estimate_initials=estimate_initials,
                                callbacks=callbacks)
    
    def forecast(self, data):
        # Make the forecas
        # Compute the Kalman filter

        n_vars = data.shape[0]
        n_measurements = data.shape[1]

        m = self.initial_value
        P = self.initial_covariance
        F = self.state_transition
        H = self.observation_model
        Q = self.process_noise
        R = self.observation_noise

        n_states = F.shape[-2]
        dim_obs = H.shape[-2]

        forecast_mean = np.empty((n_vars, n_measurements, dim_obs))
        forecast_cov = np.empty((n_vars, n_measurements, dim_obs, dim_obs))
        predicted_state_mean = np.empty((n_vars, n_measurements + 1, n_states))
        predicted_state_cov = np.empty((n_vars, n_measurements + 1, n_states, n_states))
        predicted_state_mean[:, 0, :] = m[..., 0]
        predicted_state_cov[:, 0, :, :] = P
        for i in range(n_measurements):
            # forecast of the endog var
            obs_mean, obs_cov = simdkalman.primitives.predict_observation(m, P, H, R)
            forecast_mean[:, i, :] = obs_mean[..., 0]
            forecast_cov[:, i, :, :] = obs_cov
            # update. R matrix is reshaped to be 3d, it's a requirement for the function
            y = data[:, i, ...].reshape((n_vars, dim_obs, 1))
            m, P = simdkalman.primitives.update(m, P, H, R, y)
            # predict
            m, P = simdkalman.primitives.predict(m, P, F, Q)
            predicted_state_mean[:, i + 1, :] = m[..., 0]
            predicted_state_cov[:, i + 1, :, :] = P

        return forecast_mean, forecast_cov, predicted_state_mean, predicted_state_cov