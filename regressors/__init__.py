from typing import Dict, Type
from regressors.base_regressor import BaseBayesianRegressor
from regressors.random import RandomRegressor
from regressors.averager import AveragerRegressor
from regressors.mle import MLERegressor


regressor_name_to_RegressorClass: Dict[str, Type[BaseBayesianRegressor]] = {
    "Random Regressor": RandomRegressor,
    "Averager Regressor": AveragerRegressor,
    "Maximum Likelihood Estimation": MLERegressor,
}
