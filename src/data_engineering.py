# Logging
from collections import defaultdict
import os
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time
from typing import Dict, List, Type
import cProfile

# ML libraries
import random
import numpy as np


def shuffle_data(*datas: np.ndarray) -> List[np.ndarray]:
    """Shuffle the data in the same way for all the datasets.

    Args:
        datas (np.ndarray): the data to shuffle. Each data should have the same number of samples.

    Returns:
        List[np.ndarray]: the shuffled data.
    """
    # Check that all the data have the same number of samples
    n_samples = datas[0].shape[0]
    for data in datas:
        assert (
            data.shape[0] == n_samples
        ), "The number of samples should be the same for all the datasets."

    # Shuffle the indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Shuffle the data
    shuffled_datas = [data[indices] for data in datas]

    return shuffled_datas
