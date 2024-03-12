import numpy as np
from datasets.base_dataset import BaseDataset


class ClassificationDataset(BaseDataset):

    def __init__(self, config) -> None:
        super().__init__(config)

        # Parameters of the dataset
        n_per_class = config["n_data_points"]
        n = 2 * n_per_class
        p = config["dim"]
        self.sigma = config["sigma"]

        first_class_means = config["first_class_means"]
        first_class_stddevs = config["first_class_stddevs"]
        second_class_means = config["second_class_means"]
        second_class_stddevs = config["second_class_stddevs"]

        assert len(first_class_means) == len(first_class_stddevs), "First class means and standard deviations should have the same length"
        assert len(second_class_means) == len(second_class_stddevs), "Second class means and standard deviations should have the same length"

        # Generate the beta vector for the pipeline to work, but useless for the classification
        self.beta = np.zeros(p)

        # Generate the X matrix
        X1 = np.random.normal(
            loc=first_class_means,
            scale=first_class_stddevs,
            size=(n_per_class, len(first_class_means)),
        )
        X1 = np.concatenate(
            (X1, np.random.normal(loc=0, scale=self.sigma, size=(n_per_class, p - len(first_class_means)))), axis=1
        )
        X2 = np.random.normal(
            loc=second_class_means,
            scale=second_class_stddevs,
            size=(n_per_class, len(second_class_means)),
        )
        X2 = np.concatenate(
            (X2, np.random.normal(loc=0, scale=self.sigma, size=(n_per_class, p - len(second_class_means)))), axis=1
        )
        self.X_matrix = np.concatenate((X1, X2))

        self.Y_observed = np.zeros(n)
        self.Y_observed[n_per_class:] = 1


    def get_x_data(self):
        return self.X_matrix

    def get_labels(self) -> np.ndarray:
        return self.Y_observed
    
    def get_z_data(self) -> np.ndarray:
        return self.Z_observed

    def get_beta(self) -> np.ndarray:
        return self.beta
