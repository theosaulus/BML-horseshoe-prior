# Logging
from collections import defaultdict
import os
from matplotlib import pyplot as plt
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
from src.data_engineering import shuffle_data

# Project imports
from src.time_measure import RuntimeMeter
from src.utils import try_get, try_get_seed
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
    val_proportion: float = try_get(config, "val_proportion", 0)
    do_val: bool = config["do_val"] and val_proportion > 0
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_plot = config["do_plot"]
    do_cli: bool = config["do_cli"]
    do_tqdm: bool = config["do_tqdm"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
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
    os.makedirs(f"logs/{run_name}", exist_ok=True)
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
            # Create the dataset and get the data
            dataset = DatasetClass(config["dataset"]["config"])
            x_data = dataset.get_x_data()  # (n, p)
            y_data = dataset.get_labels()  # (n,)
            # Shuffle the data
            x_data, y_data = shuffle_data(x_data, y_data)
            # Split the data into train and val
            if do_val:
                n_val = int(val_proportion * x_data.shape[0])
                assert (
                    n_val > 0
                ), "The proportion of validation data should be greater than 0."
                x_val, x_train = x_data[:n_val], x_data[n_val:]
                y_val, y_train = y_data[:n_val], y_data[n_val:]

        # Get the regressor result, and measure the time.
        with RuntimeMeter("regressor") as rm:
            beta_hat = regressor.find_coefficients(x_train, y_train)

        # Compute metrics
        metric_result = {}
        with RuntimeMeter("metric") as rm:
            beta = dataset.get_beta()  # (p,)

            # Compute estimation metrics (i.e. for the task of estimating beta)
            metric_result["estimation/l2_error"] = np.mean((beta - beta_hat) ** 2)
            metric_result["estimation/l1_error"] = np.mean(np.abs(beta - beta_hat))
            metric_result["estimation/l_inf_error"] = np.max(np.abs(beta - beta_hat))

            # Compute train prediction metrics (i.e. for the task of predicting y)
            y_pred_train = x_train @ beta_hat
            metric_result["prediction/l2_error_train"] = np.mean(
                (y_train - y_pred_train) ** 2
            )
            metric_result["prediction/l1_error_train"] = np.mean(
                np.abs(y_train - y_pred_train)
            )
            metric_result["prediction/l_inf_error_train"] = np.max(
                np.abs(y_train - y_pred_train)
            )
            # Compute sigma2 normalized metrics
            if hasattr(dataset, "sigma"):
                residuals_averager = np.ones(y_train.shape) * dataset.sigma
            else:
                residuals_averager = np.mean(y_train) - y_pred_train
            metric_result["prediction_normalized/l2_error_train"] = metric_result[
                "prediction/l2_error_train"
            ] / np.mean(residuals_averager**2)
            metric_result["prediction_normalized/l1_error_train"] = metric_result[
                "prediction/l1_error_train"
            ] / np.mean(np.abs(residuals_averager))
            metric_result["prediction_normalized/l_inf_error_train"] = metric_result[
                "prediction/l_inf_error_train"
            ] / np.max(np.abs(residuals_averager))

            if do_val:
                # Compute val prediction metrics
                y_pred_val = x_val @ beta_hat
                metric_result["prediction/l2_error_val"] = np.mean(
                    (y_val - y_pred_val) ** 2
                )
                metric_result["prediction/l1_error_val"] = np.mean(
                    np.abs(y_val - y_pred_val)
                )
                metric_result["prediction/l_inf_error_val"] = np.max(
                    np.abs(y_val - y_pred_val)
                )
                # Compute sigma2 normalized metrics
                if hasattr(dataset, "sigma"):
                    residuals_averager = np.ones(y_val.shape) * dataset.sigma
                else:
                    residuals_averager = np.mean(y_val) - y_pred_val
                metric_result["prediction_normalized/l2_error_val"] = metric_result[
                    "prediction/l2_error_val"
                ] / np.mean(residuals_averager**2)
                metric_result["prediction_normalized/l1_error_val"] = metric_result[
                    "prediction/l1_error_val"
                ] / np.mean(np.abs(residuals_averager))
                metric_result["prediction_normalized/l_inf_error_val"] = metric_result[
                    "prediction/l_inf_error_val"
                ] / np.max(np.abs(residuals_averager))
                # Plot y_pred = f(y) for the first dataset
                if do_plot and idx_dataset == 0:
                    if hasattr(dataset, "sigma"):
                        plt.scatter(y_train / dataset.sigma, y_pred_train / dataset.sigma, label="y_pred_train / sigma")
                        if do_val:
                            plt.scatter(y_val / dataset.sigma, y_pred_val / dataset.sigma, label="y_pred_val / sigma")
                        plt.xlabel("y (unit of sigma)")
                        plt.ylabel("y_pred (unit of sigma)")
                    else:
                        plt.scatter(y_train, y_pred_train, label="y_pred_train")
                        if do_val:
                            plt.scatter(y_val, y_pred_val, label="y_pred_val")
                        plt.xlabel("y")
                        plt.ylabel("y_pred")
                    plt.plot([np.min(y_data), np.max(y_data)], [np.min(y_data), np.max(y_data)], label="Flat", color="black")
                    plt.title(f"Signal detection with {predictor_name} \non dataset '{dataset_name}'")
                    plt.legend()
                    plt.savefig(f"logs/{run_name}/Signal detection.png")

            # Average those metrics over the number of datasets
            for metric_name in metric_result:
                metric_result_averaged[f"{metric_name}_averaged"] = (
                    metric_result_averaged[f"{metric_name}_averaged"] * idx_dataset
                    + metric_result[metric_name]
                ) / (idx_dataset + 1)

            # Merge the averaged metrics with the current metrics
            metric_result_all = {**metric_result, **metric_result_averaged}

            # Add runtime metrics and other metrics
            metric_result_all.update(
                {
                    f"runtime/runtime_{stage_name}": stage_runtime
                    for stage_name, stage_runtime in rm.get_stage_runtimes().items()
                }
            )
            metric_result_all["runtime/total_runtime"] = rm.get_total_runtime()
            metric_result_all["other/idx_dataset"] = idx_dataset
            metric_result_all["other/X_matrix_total_variance"] = (
                dataset.compute_total_variance(x_train)
            )

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
                    f"\nMetric results at iteration {idx_dataset} : {metric_result_all}"
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
