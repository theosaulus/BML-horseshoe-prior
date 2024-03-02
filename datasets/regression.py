import numpy as np
from datasets.base_dataset import BaseDataset


class RegressionDataset(BaseDataset):

    def __init__(self, config) -> None:
        super().__init__(config)

        # Parameters of the dataset
        n = config["n_data_points"]
        p = config["dim"]
        sigma = config["sigma"]
        feature_correlation = config["feature_correlation"]
        data_point_correlation_factor = config["data_point_correlation_factor"]

        # Generate the beta vector
        self.beta = self.generate_beta(
            p=p,
            non_zero_first_coefficients=config["non_zero_first_coefficients"],
            non_zero_proportion=config["non_zero_proportion"],
            non_zero_proportion_min=config["non_zero_proportion_min"],
            non_zero_proportion_max=config["non_zero_proportion_max"],
            non_zero_student_t_degree_freedom=config[
                "non_zero_student_t_degree_freedom"
            ],
            non_zero_student_t_scale=config["non_zero_student_t_scale"],
        )

        # Define the data covariance matrix S = (s_ij) where s_ij = 1 if i = j and s_ij = feature_correlation otherwise
        covariance_matrix = np.ones((p, p)) * feature_correlation
        covariance_matrix[np.diag_indices(p)] = 1

        # Define the mean of the unnormlized data
        mean_x_unormalized = np.zeros((p,))
        mean_x_unormalized[0] = data_point_correlation_factor

        # Generate the unnormalized data
        self.X_matrix = np.random.multivariate_normal(
            mean=mean_x_unormalized,
            cov=covariance_matrix,
            size=n,
        )

        # Normalize the data
        self.X_matrix = (
            self.X_matrix / np.sqrt(np.sum(self.X_matrix**2, axis=1))[:, np.newaxis]
        )

        # Generate the observed labels
        self.Y_observed = np.random.normal(
            loc=self.X_matrix @ self.beta,
            scale=sigma,
        )

    def get_x_data(self):
        return self.X_matrix

    def get_labels(self) -> np.ndarray:
        return self.Y_observed

    def get_beta(self) -> np.ndarray:
        return self.beta
