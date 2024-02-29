from tkinter import Y
import numpy as np
from datasets.base_dataset import BaseDataset


class MeanDataset(BaseDataset):

    def __init__(self, config) -> None:
        super().__init__(config)

        # Parameters of the dataset
        self.p = config["dim"]
        self.n = self.p  # The number of samples is equal to the dimension of the data for the mean dataset
        self.non_zero_student_t_scale = config["non_zero_student_t_scale"]
        self.non_zero_student_t_degree_freedom = config[
            "non_zero_student_t_degree_freedom"
        ]
        self.sigma = config["sigma"]

        # Generate the beta vector
        if config["non_zero_first_coefficients"] is not None:
            non_zero_first_coefficients = config["non_zero_first_coefficients"]
            assert (
                len(non_zero_first_coefficients) <= self.p
            ), "The number of non-zero coefficients should be less than the dimension of the data."
            self.beta = np.zeros(self.p)
            self.beta[: len(non_zero_first_coefficients)] = non_zero_first_coefficients

        else:
            if config["non_zero_proportion"] is not None:
                non_zero_proportion = config["non_zero_proportion"]
            else:
                non_zero_proportion = np.random.uniform(
                    config["non_zero_proportion_min"], config["non_zero_proportion_max"]
                )
            self.beta = np.zeros(shape=(self.p,))
            for j in range(self.p):
                if np.random.rand() < non_zero_proportion:
                    self.beta[j] = (
                        np.random.standard_t(
                            self.non_zero_student_t_degree_freedom,
                        )
                        * self.non_zero_student_t_scale
                    )
                    
        # Generate the data
        self.X_matrix = np.eye(self.p)
        self.Y_observed = np.random.normal(
            loc=self.beta,
            scale=self.sigma,
        )

    def get_x_data(self):
        return self.X_matrix

    def get_labels(self) -> np.ndarray:
        return self.Y_observed
    
    def get_beta(self) -> np.ndarray:
        return self.beta
