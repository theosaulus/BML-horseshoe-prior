from typing import Dict, Type
from predictors.base_predictor import BasePredictor
from predictors.random import RandomPredictor
from predictors.averager import AveragerPredictor
from predictors.mle import MLEPredictor


solver_name_to_SolverClass: Dict[str, Type[BasePredictor]] = {
    "Random Predictor": RandomPredictor,
    "Averager Predictor": AveragerPredictor,
    "Maximum Likelihood Estimation": MLEPredictor,
}
