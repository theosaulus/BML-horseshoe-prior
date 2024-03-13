from typing import Dict, List
import numpy as np
import pymc as pm
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from regressors.base_regressor import BaseBayesianRegressor
from regressors.bayesian_lm import BaseBayesianLinearPM

class BayesianLinearHyperlassoPM(BaseBayesianLinearPM):
    """ A regressor that uses the PyMC library to perform Bayesian linear regression with different priors.
    The priors available are the horseshoe, laplacian and student-t.
    It is possible to use the Numpyro sampler by setting the `use_numpyro` parameter to True (requires Jax to be installed).
    MCMC sampling is done by default or variational inference by setting the `use_vi` parameter to True.
    For MCMC sampling, the NUTS sampler is used by default.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        assert self.prior_name in ["horseshoe", "reg-horseshoe", "laplacian", "student", "hyperlasso"], "Invalid prior. Choose from 'horseshoe', 'reg-horseshoe', 'laplacian', or 'student'."
        
    def find_coefficients(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        hyperlasso = pm.Model()

        self.p = x_data.shape[1]
        p = self.p

        with hyperlasso:
            sigma = pm.HalfNormal("sigma", 1)
            
            lambda_ = pm.HalfCauchy('lambda_', beta=1, shape=(p, 1))
            tau = pm.Gamma('tau', alpha=0.5, beta=1/lambda_**2, shape=(p, 1))
            phi_sq = pm.Exponential('phi_sq', lam=tau, shape=(p, 1))

            kappa = pm.Deterministic('kappa', 1/(1+phi_sq))

            z = pm.Normal('z', mu=0, sigma=1, shape=(p, 1))
            beta = pm.Deterministic('beta', z*phi_sq)

            mu = pm.math.dot(x_data, beta)

            if self.classification:
                likelihood = pm.invlogit(mu)
                y_obs = pm.Bernoulli('y_obs', p=likelihood, observed=y_data.reshape(-1, 1))
            else:
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data.reshape(-1, 1))

            self.prior = pm.sample_prior_predictive(samples=1000)
            if self.use_vi:
                approx = pm.fit(10000, method="advi", random_seed=self.seed)
                trace = approx.sample(10000, random_seed=self.rng)
            else:
                if self.use_numpyro:
                    trace = pm.sample(10000, nuts_sampler="numpyro", target_accept=0.8, random_seed=self.rng)
                else:
                    trace = pm.sample(10000, target_accept=0.8, random_seed=self.rng, chains=self.chains)
        
        self.trace = trace

        betas_sampled = trace["posterior"]["beta"].mean(axis=0)

        beta_hat = betas_sampled.mean(axis=0)[:, 0]
        return beta_hat.values