from typing import Dict, List

import numpy as np
from regressors.base_regressor import BaseBayesianRegressor


class AveragerRegressor(BaseBayesianRegressor):

    def __init__(self, config: Dict):
        super().__init__(config)
    
    def find_coefficients(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Averager regressor not implemented yet.")
