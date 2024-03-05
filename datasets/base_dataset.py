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

    # ========== Helper functions ==========

    def generate_beta(
        self,
        p: int,
        non_zero_first_coefficients: Union[None, np.ndarray],
        non_zero_proportion: Union[None, float],
        non_zero_proportion_min: float,
        non_zero_proportion_max: float,
        non_zero_student_t_degree_freedom: float,
        non_zero_student_t_scale: float,
    ) -> np.ndarray:
        """Generate the beta vector following the configuration.

        Args:
            p (int): the dimension of each data point
            non_zero_first_coefficients (Union[None, np.ndarray], optional): the first coefficients that are non-zero.
            non_zero_proportion (Union[None, float], optional): the proportion of non-zero coefficients (if non_zero_first_coefficients is None)
            non_zero_proportion_min (float, optional): the minimum proportion of non-zero coefficients (if non_zero_proportion is None)
            non_zero_proportion_max (float, optional):  the maximum proportion of non-zero coefficients (if non_zero_proportion is None)
            non_zero_student_t_degree_freedom (float, optional): the degree of freedom of the student-t distribution for the non-zero coefficients
            non_zero_student_t_scale (float, optional): the scale of the student-t distribution for the non-zero coefficients

        Returns:
            np.ndarray: the beta vector. Shape (p,).
        """
        # If non_zero_first_coefficients is not None, then the first coefficients are the specified coefficients and the rest are zero
        if non_zero_first_coefficients is not None:
            assert (
                len(non_zero_first_coefficients) <= p
            ), "The number of non-zero coefficients should be less than the dimension of the data."
            beta = np.zeros((p,))
            beta[: len(non_zero_first_coefficients)] = non_zero_first_coefficients

        # Else, the proportion of non-zero coefficients is either specified or randomly generated
        else:
            # Randomly generate the proportion of non-zero coefficients if not specified
            if non_zero_proportion is None:
                non_zero_proportion = np.random.uniform(
                    non_zero_proportion_min, non_zero_proportion_max
                )

            # Sample each non-zero coefficient from a student-t distribution
            beta = np.zeros(shape=(p,))
            for j in range(p):
                if np.random.rand() < non_zero_proportion:
                    beta[j] = (
                        np.random.standard_t(
                            non_zero_student_t_degree_freedom,
                        )
                        * non_zero_student_t_scale
                    )

        return beta

    def compute_total_variance(self, X_matrix: np.ndarray) -> np.ndarray:
        """Compute the total variance of the data.
        It is defined as the sum of the variance of each feature.

        Args:
            X_matrix (np.ndarray): the data. Shape (n, p).

        Returns:
            float : the average variance of the data.
        """
        return np.sum(np.var(X_matrix, axis=0))
