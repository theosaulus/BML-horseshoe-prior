from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class BaseDataset(ABC):
    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def get_x_data(self) -> np.ndarray:
        """Get the input data.

        Returns:
            np.ndarray: the input data. Shape (n, p).
        """     

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """Get the labels.
        
        Returns:
            np.ndarray: the labels. Shape (n,).
        """

    @abstractmethod
    def get_beta(self) -> np.ndarray:
        """Get the true coefficients.
        
        Returns:
            np.ndarray: the true coefficients. Shape (p,).
        """