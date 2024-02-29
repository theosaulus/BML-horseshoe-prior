from typing import Dict, List

import numpy as np
from predictors.base_predictor import BasePredictor


class AveragerPredictor(BasePredictor):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.slope = config["slope"]
        self.intercept = config["intercept"]

    def fit(self, x_data: np.ndarray) -> Dict[int, List[int]]:
        return self.slope * x_data + self.intercept
