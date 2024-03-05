from typing import Dict, List
import numpy as np
import pymc as pm
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from regressors.base_regressor import BaseBayesianRegressor

class BayesianLinearPM(BaseBayesianRegressor):
    """ A regressor that uses the PyMC library to perform Bayesian linear regression with different priors.
    The priors available are the horseshoe, laplacian and student-t.
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
        self.prior = config.get("prior", "horseshoe")
        assert self.prior in ["horseshoe", "laplacian", "student"], "Invalid prior. Choose from 'horseshoe', 'laplacian' or 'student'."
        self.rng = np.random.default_rng(self.seed)
        
    def find_coefficients(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        horseshoe = pm.Model()

        self.p = x_data.shape[1]
        p = self.p

        with horseshoe:
            sigma = pm.HalfNormal("sigma", 1)

            tau = pm.HalfCauchy('tau', beta=1)

            if self.prior == "horseshoe":
                lambda_ = pm.HalfCauchy('lambda_', beta=1, shape=(p, 1))
            elif self.prior == "laplacian":
                lambda_sq = pm.Exponential('lambda_sq', lam=2, shape=(p, 1))
                lambda_ = pm.Deterministic('lambda_', pm.math.sqrt(lambda_sq))
            elif self.prior == "student":
                lambda_sq = pm.InverseGamma('lambda_sq', alpha=1, beta=2, shape=(p, 1))
                lambda_ = pm.Deterministic('lambda_', pm.math.sqrt(lambda_sq))

            kappa = pm.Deterministic('kappa', 1/(1+lambda_**2))

            z = pm.Normal('z', mu=0, sigma=1, shape=(p, 1))
            beta = pm.Deterministic('beta', z*tau*lambda_)

            mu = pm.math.dot(x_data, beta)

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
    
    def plot_posterior(self, true_post=None, var_name="beta", ax=None, title=r"$\beta$ Posterior", x_label=r"$\beta$"):
        ax, = az.plot_forest(self.trace, var_names=[var_name], coords={"beta_dim_0": range(len(self.trace.posterior.beta_dim_0))},
            kind='ridgeplot', ridgeplot_truncate=False, ridgeplot_alpha=0.5,
            hdi_prob=0.95, combined=True, ax=ax,
            figsize=(8, 6))

        ax.set_title(title)
        ax.set_xlabel(r"$\beta$")

        if true_post is not None:
            ax.scatter(true_post[::-1], ax.get_yticks(), color='r', marker='x', s=100, label='True beta')
        
        return ax
    
    def plot_prior(self, var_name="beta", ax=None, title=r"$\beta$ Prior", x_label=r"$\beta$", xmin=-10, xmax=10):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        prior_plot = self.prior.prior[var_name].mean(axis=0).values.reshape(-1)
        if xmin is not None:
            prior_plot = prior_plot[prior_plot > xmin]
        if xmax is not None:
            prior_plot = prior_plot[prior_plot < xmax]

        ax = sns.kdeplot(prior_plot, levels=30, alpha=0.5, ax=ax)
        # plt.xlim(-100, 100)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Density")
        ax.set_xlim(xmin, xmax)

        return ax