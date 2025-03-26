# BML-horseshoe-prior

A benchmark of several bayesian estimators on sparse linear regression problems, with a focus on the horseshoe prior and the article Carvalho, Carlos M., Nicholas G. Polson, and James G. Scott. "Handling sparsity via the horseshoe." Artificial intelligence and statistics. PMLR, 2009.

This is a student project by Timothé Boulet, Ali Ramlaoui and Théo Saulus, for the course "Bayesian Machine Learning" at the MVA master.


# Installation

Clone the repository, create a venv (advised), and install the requirements:

```bash
git clone git@github.com:theosaulus/BML-horseshoe-prior.git
cd BML-horseshoe-prior
python -m venv venv
source venv/bin/activate  # on linux
venv\Scripts\activate  # on windows
pip install -r requirements.txt
```


# Run the code
 
The framework is divided in two parts : the dataset (the task to solve) and the regressor (the Bayesian estimator you will use to solve the task). You can fit any regressor on any dataset, and compare the results with the metrics.

For fitting regressor of tag ``<regressor tag>`` on dataset of tag ``<dataset tag>``, run the following command:

```bash
python run.py regressor=<regressor tag> dataset=<dataset tag>
```

For example, to fit a random regressor (`random`) on the task of sparse mean estimation (`mean`), run the following command:

```bash
python run.py regressor=random dataset=mean
```

We use Hydra as our config system. The config folder is `./configs/`. You can modify the config (logging, metrics, number of datasets) from the `default_config.yaml` file. You can also create your own config file and specify it with the `--config-name` argument :

```bash
python run.py regressor=random dataset=mean --config-name=my_config_name
```

# Regressors
The algo tag should correspond to a configuration in ``configs/regressor/`` where you can specify the regressor and its hyperparameters. 

Currently, the following regressors are available:
 - `random` : Randomly predict the $\beta$ parameter vector.
 - `mle` : Maximum likelihood estimation, by doing a linear regression with the least squares method.
 - `laplacian` : Laplacian prior, corresponding to a LASSO.
 - `student` : Student-t prior.
 - `horseshoe` : Horseshoe prior.
 - `reg-horseshoe` : Regularized horseshoe prior.
 - `spike_slab` : Spike and slab prior.
 - `hyperlasso` : Hyperlasso prior.

# Datasets

The dataset tag should correspond to a configuration in ``configs/dataset/`` where you can specify the dataset and its hyperparameters.

Currently the following datasets are available:
 - `mean` : Sparse mean estimation, i.e. the task of estimating the mean of a sparse vector from noisy observations.
 - `reg` : Sparse linear regression, i.e. the task of estimating the coefficients of a linear model from noisy observations.
 - `classification` : Sparse classification, i.e. the task of estimating the coefficients of a logistic regression model from noisy observations.
 - `exp1` : First experiment of the paper.
 - `exp2` : Second experiment of the paper.
 - `exp3` : Third experiment of the paper.


# Results and visualization

### Report

Our course report is accessible in the file `report.pdf`. Link : [report.pdf](report.pdf).

Our conclusion :

    We acknowledge the clear contribution of the paper, which is to apply the horseshoe prior to supervised learning tasks, while proposing explanations for its shrinkage power. 

    We tried to address what appeared as weak points to us, namely using an actually fully Bayesian framework instead of plug-in during the experiments, providing more comprehensive comparisons with other priors, and doing an experiment on a task designed to challenge the prior. 
    
    More specifically, we notice that in many experiments, the Student-t prior approximately matches the performances of the horseshoe, which was not obvious when reading the article. Thus, although the horseshoe (or its regularized version) is a very efficient and flexible prior, it may not perform significantly different from Student-t, which should have been mentioned in the first place.

### WandB
WandB is a very powerful tool for logging. It is flexible, logs everything online, can be used to compare experiments or group those by dataset or algorithm, etc. You can also be several people to work on the same project and share the results directly on line. It is also very easy to use, and can be used with a few lines of code.

If `do_wandb` is True, the metrics will be logged in the project `wandb_config['project']`, and you can visualize the results on the WandB website.

### Tensorboard
Tensorboard is a tool that allows to visualize the training. It is usefull during the development phase, to check that everything is working as expected. It is also very easy to use, and can be used with a few lines of code.

If `do_tb` is True, you can visualize the logs by running the following command in the terminal.
```bash
tensorboard --logdir=tensorboard
```

### CLI

You can also visualize the results in the terminal. If `do_cli` is True, the metrics will be printed in the terminal every `cli_frequency_episode` episodes.

# Other

### Seed

You can specify the seed of the experiment with the `seed` argument. If you don't specify it, the seed will be randomly chosen.

### cProfile and SnakeViz

cProfile is a module that allows to profile the code. It is very useful to find bottlenecks in the code, and to optimize it. SnakeViz is a tool that allows to visualize the results of cProfile and so what you should focus. It is used through the terminal :

```bash
snakeviz logs/profile_stats.prof
```