import numpy as np
from datasets.base_dataset import BaseDataset


class RegressionDataset(BaseDataset):

    def __init__(self, config) -> None:
        super().__init__(config)
        raise NotImplementedError
