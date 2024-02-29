from typing import Dict, List

import numpy as np
from regressors.base_regressor import BaseBayesianRegressor
from sklearn.linear_model import LinearRegression

class MLERegressor(BaseBayesianRegressor):
    """ A regressor that uses the maximum likelihood estimation to find the coefficients.
    For regression under a gaussian model, this is equivalent to the least squares method.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
    def find_coefficients(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        model = LinearRegression().fit(x_data, y_data)
        return model.coef_