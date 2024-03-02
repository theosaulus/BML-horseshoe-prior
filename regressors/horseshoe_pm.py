from typing import Dict, List
import numpy as np
import pymc as pm
from regressors.base_regressor import BaseBayesianRegressor

class HorseshoePM(BaseBayesianRegressor):
    """ A regressor that uses the Horseshoe prior implemented in PyMC3.
    It is possible to use the Numpyro sampler by setting the `use_numpyro` parameter to True (requires Jax to be installed).
    MCMC sampling is done by default or variational inference by setting the `use_vi` parameter to True.
    For MCMC sampling, the NUTS sampler is used by default.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.use_numpyro = config.get("use_numpyro", False)
        self.use_vi = config.get("use_vi", False)
        self.chains = config.get("chains", 4)
        self.seed = config.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        
    def find_coefficients(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        horseshoe = pm.Model()

        p = x_data.shape[1]

        with horseshoe:
            sigma = pm.HalfNormal("sigma", 1)

            tau = pm.HalfCauchy('tau', beta=1)
            lambda_ = pm.HalfCauchy('lambda_', beta=1, shape=(p, 1))

            z = pm.Normal('z', mu=0, sigma=1, shape=(p, 1))
            beta = pm.Deterministic('beta', z*tau*lambda_)

            mu = pm.math.dot(x_data, beta)

            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data.reshape(-1, 1))

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