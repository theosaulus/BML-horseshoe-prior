# Logging
from collections import defaultdict
import os
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict

# Utils
from tqdm import tqdm
import datetime
from time import time
from typing import Dict, List, Type
import cProfile

# ML libraries
import random
import numpy as np

# Project imports
from src.time_measure import RuntimeMeter
from src.utils import try_get_seed
from datasets import dataset_name_to_DatasetClass
from regressors import regressor_name_to_RegressorClass


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))

    # Get the config values from the config object.
    predictor_name: str = config["regressor"]["name"]
    dataset_name: str = config["dataset"]["name"]
    n_datasets: int = config["n_datasets"]
    do_cli: bool = config["do_cli"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_tqdm: bool = config["do_tqdm"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    with open_dict(config):
        config.regressor.config.seed = seed
    print(f"Using seed: {seed}")

    # Get the regressor
    print("Creating the regressor...")
    RegressorClass = regressor_name_to_RegressorClass[predictor_name]
    regressor = RegressorClass(config=config["regressor"]["config"])

    # Create the dataset class
    print("Creating the dataset...")
    DatasetClass = dataset_name_to_DatasetClass[dataset_name]

    # Initialize loggers
    run_name = f"[{predictor_name}]_[{dataset_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{np.random.randint(seed)}"
    print(f"\nStarting run {run_name}")
    os.makedirs("logs", exist_ok=True)
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=OmegaConf.to_container(config),
            **config["wandb_config"],
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")
    metric_result_averaged: Dict[str, float] = defaultdict(float)

    # Training loop
    for idx_dataset in tqdm(range(n_datasets), disable=not do_tqdm):

        # Load data
        with RuntimeMeter("dataset") as rm:
            dataset = DatasetClass(config["dataset"]["config"])
            x_data = dataset.get_x_data()  # (n, p)
            y_data = dataset.get_labels()  # (n,)

        # Get the regressor result, and measure the time.
        with RuntimeMeter("regressor") as rm:
            beta_hat = regressor.find_coefficients(x_data, y_data)

        # Compute metrics
        metric_result = {}
        with RuntimeMeter("metric") as rm:
            beta = dataset.get_beta()  # (p,)

            # Compute estimation metrics (i.e. for the task of estimating beta)
            metric_result["estimation/l2_error"] = np.mean((beta - beta_hat) ** 2)
            metric_result["estimation/l1_error"] = np.mean(np.abs(beta - beta_hat))
            metric_result["estimation/l_inf_error"] = np.max(np.abs(beta - beta_hat))

            # Compute prediction metrics (i.e. for the task of predicting y)
            y_hat = x_data @ beta_hat
            metric_result["prediction/l2_error"] = np.mean((y_data - y_hat) ** 2)
            metric_result["prediction/l1_error"] = np.mean(np.abs(y_data - y_hat))
            metric_result["prediction/l_inf_error"] = np.max(np.abs(y_data - y_hat))
            # Eventually compute sigma2 normalized metrics
            if hasattr(dataset, "sigma"):
                residuals_averager = np.ones(y_data.shape) * dataset.sigma
            else:
                residuals_averager = np.mean(y_data) - y_hat
            metric_result["prediction/l2_error_normalized"] = metric_result[
                "estimation/l2_error"
            ] / np.mean(residuals_averager**2)
            metric_result["prediction/l1_error_normalized"] = metric_result[
                "estimation/l1_error"
            ] / np.mean(np.abs(residuals_averager))
            metric_result["prediction/l_inf_error_normalized"] = metric_result[
                "estimation/l_inf_error"
            ] / np.max(np.abs(residuals_averager))

            # Average those metrics over the number of datasets
            for metric_name in metric_result:
                metric_result_averaged[f"{metric_name}_averaged"] = (
                    metric_result_averaged[f"{metric_name}_averaged"] * idx_dataset
                    + metric_result[metric_name]
                ) / (idx_dataset + 1)

            # Merge the averaged metrics with the current metrics
            metric_result_all = {**metric_result, **metric_result_averaged}

            # Add runtime metrics and misc metrics
            metric_result_all.update(
                {
                    f"runtime/runtime_{stage_name}": stage_runtime
                    for stage_name, stage_runtime in rm.get_stage_runtimes().items()
                }
            )
            metric_result_all["runtime/total_runtime"] = rm.get_total_runtime()
            metric_result_all["idx_dataset"] = idx_dataset

        # Log the metrics
        with RuntimeMeter("log") as rm:
            if do_wandb:
                wandb.log(metric_result_all, step=idx_dataset)
            if do_tb:
                for metric_name, metric_value in metric_result_all.items():
                    tb_writer.add_scalar(
                        metric_name,
                        metric_value,
                        global_step=idx_dataset,
                    )
            if do_cli:
                print(
                    f"Metric results at iteration {idx_dataset} : {metric_result_all}"
                )

    # Finish the WandB run.
    if do_wandb:
        run.finish()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
