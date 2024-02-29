from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class BaseBayesianRegressor(ABC):
    """The base class for all bayesian predictors. A bayesian predictor is an object that, given x data and y data, is able
    to predict the beta coefficients of the model that best fits the data according to it's prior.
    """

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def find_coefficients(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        """Given the x_data and y_data, return the beta coefficients of the model that best fits the data according to it's prior.
        
        Args:
            x_data (np.ndarray): The x data. Shape (n, p)
            y_data (np.ndarray): The y data. Shape (n,)
            
        Returns:
            np.ndarray: The beta estimated (beta_hat) coefficients. Shape (p,)
        """
