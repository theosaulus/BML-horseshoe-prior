from typing import Dict, List
import numpy as np
import pymc as pm
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from regressors.base_regressor import BaseBayesianRegressor
from regressors.bayesian_lm import BaseBayesianLinearPM

class BayesianLinearSpikeSlabPM(BaseBayesianLinearPM):
    """ A regressor that uses the PyMC library to perform Bayesian linear regression with different priors.
    The priors available are the horseshoe, laplacian and student-t.
    It is possible to use the Numpyro sampler by setting the `use_numpyro` parameter to True (requires Jax to be installed).
    MCMC sampling is done by default or variational inference by setting the `use_vi` parameter to True.
    For MCMC sampling, the NUTS sampler is used by default.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
    def find_coefficients(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        spike_slab = pm.Model()

        self.p = x_data.shape[1]
        p = self.p

        with spike_slab:
            sigma = pm.HalfNormal("sigma", 1)

            c = 5 # TODO, allow to have a prior for c
            epsilon = 0.1

            lambda_ = pm.Bernoulli('lambda_', p=0.5, shape=(p, 1))

            kappa = pm.Deterministic('kappa', 1/(1+lambda_**2))

            if epsilon == 0:
                z = pm.Normal('z', mu=0, sigma=c, shape=(p, 1))
                beta = pm.Deterministic('beta', z*lambda_)
            else:
                z_1 = pm.Normal('z_1', mu=0, sigma=epsilon, shape=(p, 1))
                z_2 = pm.Normal('z_2', mu=0, sigma=c, shape=(p, 1))
                beta = pm.Deterministic('beta', lambda_*z_1 + (1-lambda_)*z_2)

            mu = pm.math.dot(x_data, beta)

            if self.classification:
                y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y_data)
            else:
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data.reshape(-1, 1))

            self.prior = pm.sample_prior_predictive(samples=1000)
            if self.use_vi:
                approx = pm.fit(10000, method="advi", random_seed=self.seed)
                trace = approx.sample(1000, random_seed=self.rng)
            else:
                if self.use_numpyro:
                    trace = pm.sample(1000, nuts_sampler="numpyro", target_accept=0.8, random_seed=self.rng)
                else:
                    trace = pm.sample(1000, target_accept=0.8, random_seed=self.rng, chains=self.chains)
        
        self.trace = trace

        betas_sampled = trace["posterior"]["beta"].mean(axis=0)

        beta_hat = betas_sampled.mean(axis=0)[:, 0]
        return beta_hat.values