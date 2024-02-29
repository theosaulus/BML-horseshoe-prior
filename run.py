# Logging
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
from typing import Dict, Type
import cProfile

# ML libraries
import random
import numpy as np

# Project imports
from src.time_measure import RuntimeMeter
from src.utils import try_get_seed
from datasets import dataset_name_to_DatasetClass
from predictors import solver_name_to_SolverClass
from folder_metrics import metrics_name_to_MetricsClass


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))

    # Get the config values from the config object.
    solver_name: str = config["solver"]["name"]
    task_name: str = config["task"]["name"]
    n_iterations: int = config["n_iterations"]
    do_cli: bool = config["do_cli"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_tqdm: bool = config["do_tqdm"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")

    # Get the solver
    print("Creating the solver...")
    SolverClass = solver_name_to_SolverClass[solver_name]
    solver = SolverClass(config=config["solver"]["config"])

    # Create the dataset
    print("Creating the dataset...")
    TaskClass = dataset_name_to_DatasetClass[task_name]
    task = TaskClass(config["task"]["config"])
    x_data = task.get_x_data()

    # Create the metrics
    metrics = {
        metric_name: MetricsClass(**config["metrics"][metric_name])
        for metric_name, MetricsClass in metrics_name_to_MetricsClass.items()
    }

    # Initialize loggers
    run_name = f"[{solver_name}]_[{task_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{np.random.randint(seed)}"
    os.makedirs("logs", exist_ok=True)
    print(f"\nStarting run {run_name}")
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=config,
            **config["wandb_config"],
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")

    # Training loop
    for iteration in tqdm(range(n_iterations), disable=not do_tqdm):
        # Get the solver result, and measure the time.
        with RuntimeMeter("solver") as rm:
            y_pred = solver.fit(x_data=x_data)

        # Log metrics.
        for metric_name, metric in metrics.items():
            with RuntimeMeter("metric") as rm:
                metric_result = metric.compute_metrics(
                    task=task,
                    y_pred=y_pred,
                    algo=solver,
                )
            for stage_name, stage_runtime in rm.get_stage_runtimes().items():
                metric_result[f"runtime_{stage_name}"] = stage_runtime
            metric_result["total_runtime"] = rm.get_total_runtime()
            metric_result["iteration"] = iteration

            with RuntimeMeter("log") as rm:
                if do_wandb:
                    cumulative_solver_time_in_ms = int(
                        rm.get_stage_runtime("solver") * 1000
                    )
                    wandb.log(metric_result, step=cumulative_solver_time_in_ms)
                if do_tb:
                    for metric_name, metric_result in metric_result.items():
                        tb_writer.add_scalar(
                            f"metrics/{metric_name}",
                            metric_result,
                            global_step=rm.get_stage_runtime("solver"),
                        )
                if do_cli:
                    print(
                        f"Metric results at iteration {iteration} for metric {metric_name}: {metric_result}"
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
