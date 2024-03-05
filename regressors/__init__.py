from typing import Dict, Type
from regressors.base_regressor import BaseBayesianRegressor
from regressors.random import RandomRegressor
from regressors.averager import AveragerRegressor
from regressors.mle import MLERegressor
from regressors.bayesian_lm import BayesianLinearPM


regressor_name_to_RegressorClass: Dict[str, Type[BaseBayesianRegressor]] = {
    "Random Regressor": RandomRegressor,
    "Averager Regressor": AveragerRegressor,
    "Maximum Likelihood Estimation": MLERegressor,
    "Bayesian Linear Model Prior": BayesianLinearPM,
}
