from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class BaseDataset(ABC):
    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def get_x_data(self) -> np.ndarray:
        """Example of abstract method."""

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """Example of abstract method."""
