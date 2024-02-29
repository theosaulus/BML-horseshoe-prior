from typing import Dict, List

import numpy as np
from predictors.base_predictor import BasePredictor


class MLEPredictor(BasePredictor):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.a = config["a"]
        self.b = config["b"]
        self.c = config["c"]

    def fit(self, x_data: np.ndarray) -> Dict[int, List[int]]:
        return self.a * x_data**2 + self.b * x_data + self.c
