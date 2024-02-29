from typing import Dict, List

import numpy as np
from regressors.base_regressor import BaseBayesianRegressor


class RandomRegressor(BaseBayesianRegressor):
    """A dummy regressor that simply returns random coefficients in [0, 1].
    This is useful for testing purposes.
    """
    def __init__(self, config: Dict):
        super().__init__(config)

    def find_coefficients(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        n, p = x_data.shape
        return np.random.randn(p)
